#!/usr/bin/env python3
"""Generate ASCII diagrams of the cov-shuffle algorithm used by the shfl
GPU kernel variants in SCAMP_VARIANT_TUPLES.

Each frame shows, for one row of work inside a tile:
  - the BEFORE state: which diagonal each lane's cov[i] slot is currently
    tracking;
  - the cov UPDATE (in-place, advances cov along its diagonal -- slot
    identity unchanged);
  - the SHUFFLE (cross-warp via smem hand-off) + the within-lane right-
    shift;
  - the AFTER state, with masked-but-still-present junk cells in (parens).

The "diagonal" labels are block-local diagonal indices (column - row in
block-local coordinates). The block has lanes 0..BLOCKSZ-1; each lane T
owns columns [T*DPT, T*DPT + DPT - 1]. At row r, lane T's cov[i]
TRACKS diagonal (T*DPT + i - r). A slot is INVALID (gets masked from
distc/distr) when its diagonal is to the left of the block's meta-
diagonal range, i.e. (T*DPT + i - r) < 0. Only lane 0 of warp 0 ever
produces invalid slots within a tile, given tile_height <= 32*DPT (the
constraint enforced by the static_assert in do_tile_shfl).

A WASTE SUMMARY at the end of each diagram counts masked cells across
the whole tile (not just the displayed rows).

Run as:
    python3 cov_shuffle_diagram.py [--blocksz N] [--dpt N] [--warp N]
                                   [--rows N]

Defaults are tiny (BLOCKSZ=8, DPT=2, WARP=4, ROWS=3) so the diagrams
fit on one screen. Realistic configs (BLOCKSZ=256, WARP=32) generate
multi-page output; pipe to less.
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

JUNK = "?"  # value held by lane 0 of warp 0 (no predecessor)


def initial_state(blocksz, dpt):
    """state[lane][slot] = diagonal-label string. At row 0, lane T cov[i] =
    cov(0, T*DPT + i) on diagonal (T*DPT + i)."""
    return [[str(t * dpt + i) for i in range(dpt)] for t in range(blocksz)]


def step_one_row(state, blocksz, dpt, warp):
    """Advance one row. Returns (new_state, handoff_writes, handoff_reads).

    Cross-warp cov hand-off: lane 31 of warp k publishes its post-update
    cov[DPT-1] to smem; lane 0 of warp k+1 reads warp k's published
    value into its cov[0] at the start of row r+1. Lane 0 of warp 0 has
    no predecessor and ends up with JUNK in cov[0]; the slot validity
    mask catches it before it can corrupt distc/distr.
    """
    new_state = [[None] * dpt for _ in range(blocksz)]
    handoff_writes = {}
    handoff_reads = {}

    warps_per_block = blocksz // warp

    for wid in range(warps_per_block):
        lane31 = wid * warp + (warp - 1)
        handoff_writes[wid] = state[lane31][dpt - 1]
    for wid in range(1, warps_per_block):
        handoff_reads[wid] = handoff_writes[wid - 1]

    for warp_id in range(warps_per_block):
        for lane_in_warp in range(warp):
            T = warp_id * warp + lane_in_warp
            for i in range(dpt - 1, 0, -1):
                new_state[T][i] = state[T][i - 1]
            if lane_in_warp > 0:
                pred = warp_id * warp + (lane_in_warp - 1)
                new_state[T][0] = state[pred][dpt - 1]
            else:
                if warp_id > 0:
                    new_state[T][0] = handoff_reads[warp_id]
                else:
                    new_state[T][0] = JUNK

    return new_state, handoff_writes, handoff_reads


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------

def slot_validity(blocksz, dpt, row):
    """mask[lane][slot] == True iff the slot is INVALID at `row` (masked
    from distc/distr). Lane T slot i invalid iff T*DPT + i < row."""
    mask = [[False] * dpt for _ in range(blocksz)]
    for T in range(blocksz):
        for i in range(dpt):
            if T * dpt + i < row:
                mask[T][i] = True
    return mask


# ---------------------------------------------------------------------------
# Width / format helpers
# ---------------------------------------------------------------------------

def compute_width(blocksz, dpt, warp, rows):
    """Find the max cell width across ALL frames (state values, lane
    headers, masked-paren variants) so every line in the diagram aligns."""
    width = 0
    state = initial_state(blocksz, dpt)
    for r in range(rows + 1):
        mask = slot_validity(blocksz, dpt, r)
        for T in range(blocksz):
            width = max(width, len("L" + str(T)))
            for i in range(dpt):
                v = str(state[T][i])
                width = max(width, len(v))
                if mask[T][i]:
                    width = max(width, len("(" + v + ")"))
        if r < rows:
            state, _, _ = step_one_row(state, blocksz, dpt, warp)
    return width


def fmt_cell(value, masked, width):
    """Right-justified cell label, parenthesized if masked."""
    if masked:
        return ("(" + str(value) + ")").rjust(width)
    return str(value).rjust(width)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PREFIX = "    "  # uniform row prefix; gives every line the same indent
LABEL_W = 14     # width of the "warp N / slot N" label column on the left


def render_state(state, mask, blocksz, dpt, warp, width, label=None):
    warps_per_block = blocksz // warp
    out = []
    if label:
        out.append(PREFIX + label)
    for warp_id in range(warps_per_block):
        head = ("warp " + str(warp_id)).ljust(LABEL_W) + ":"
        for lane_in_warp in range(warp):
            T = warp_id * warp + lane_in_warp
            head += " " + ("L" + str(T)).rjust(width)
        out.append(PREFIX + head)
        for slot in range(dpt):
            row = ("  slot " + str(slot)).ljust(LABEL_W) + ":"
            for lane_in_warp in range(warp):
                T = warp_id * warp + lane_in_warp
                row += " " + fmt_cell(state[T][slot], mask[T][slot], width)
            out.append(PREFIX + row)
        if warp_id < warps_per_block - 1:
            sep_len = LABEL_W + 1 + (width + 1) * warp
            out.append(PREFIX + "-" * sep_len)
    return "\n".join(out)


def render_handoff(handoff_writes, handoff_reads, blocksz, warp, width):
    warps_per_block = blocksz // warp
    out = [PREFIX + "smem cov_handoff[2 * warps_per_block=" +
           str(warps_per_block) + "]:"]
    head = ("  warp slot").ljust(LABEL_W) + ":"
    for wid in range(warps_per_block):
        head += " " + ("w" + str(wid)).rjust(width)
    out.append(PREFIX + head)
    write_row = ("  lane31 WRITE").ljust(LABEL_W) + ":"
    for wid in range(warps_per_block):
        v = str(handoff_writes.get(wid, "-"))
        write_row += " " + v.rjust(width)
    out.append(PREFIX + write_row)
    read_row = ("  lane0  READ").ljust(LABEL_W) + ":"
    for wid in range(warps_per_block):
        v = handoff_reads.get(wid, None)
        s = "-" if v is None else str(v)
        read_row += " " + s.rjust(width)
    out.append(PREFIX + read_row)
    return "\n".join(out)


def banner(text, char="="):
    bar = char * max(72, len(text) + 4)
    return "\n".join([bar, "  " + text, bar])


# ---------------------------------------------------------------------------
# Waste model
# ---------------------------------------------------------------------------

def waste_summary(blocksz, dpt, tile_height):
    """Compute analytical waste across an ENTIRE tile (tile_height rows),
    independent of the display row count.

    Returns dict with: total_cells, masked_cells, waste_pct.

    Total compute slots per tile per block = BLOCKSZ * DPT * tile_height.
    Masked compute slots: cells where slot_validity is True at row r.

    Closed form (for tile_height <= 32*DPT):
      masked_cells = tile_height * (tile_height - 1) / 2
      waste_pct = (tile_height - 1) / (2 * BLOCKSZ * DPT) * tile_height * 100
    Only warp 0 contributes -- larger warp counts dilute the same fixed
    junk-triangle across a wider block.
    """
    total = blocksz * dpt * tile_height
    masked = 0
    for T in range(blocksz):
        for i in range(dpt):
            threshold = T * dpt + i
            invalid_rows = max(0, tile_height - 1 - threshold)
            masked += invalid_rows
    return {
        "total_cells": total,
        "masked_cells": masked,
        "waste_pct": (masked / total) * 100 if total else 0.0,
    }


def render_waste_table(blocksz, dpt, tile_heights, header_note=""):
    """Render a table: rows = tile_height, columns = waste% at this geometry."""
    out = []
    if header_note:
        out.append(PREFIX + header_note)
    out.append(PREFIX + "  tile_height      total cells   wasted")
    out.append(PREFIX + "  -----------      -----------   ------")
    for th in tile_heights:
        s = waste_summary(blocksz, dpt, th)
        out.append(PREFIX + "  {:>11}      {:>11}   {:>6.2f}%".format(
            th, s["total_cells"], s["waste_pct"]))
    return "\n".join(out)


def render_waste_summary(blocksz, dpt, warp, tile_height):
    """Show waste at the demo geometry (over a tile_height sweep) AND at
    the realistic kernel geometry (BLOCKSZ=128, warp=32, same DPT)."""
    out = []
    out.append(banner("compute-waste summary", "-"))
    out.append("")
    out.append(PREFIX + "Per-tile compute = BLOCKSZ * DPT * tile_height cells.")
    out.append(PREFIX + "Masked cells = slots where the cov is for a diagonal")
    out.append(PREFIX + "outside the block's meta-diag range; the multiply")
    out.append(PREFIX + "happens but the dist update is gated off.")
    out.append("")

    # Demo geometry sweep.
    demo_heights = sorted({1, dpt, 2 * dpt, warp * dpt // 4, warp * dpt // 2,
                           warp * dpt, 2 * warp * dpt, 4 * warp * dpt})
    demo_heights = [h for h in demo_heights if h >= 1]
    out.append(
        PREFIX + "DEMO geometry: BLOCKSZ={} DPT={} warp={} (warps_per_block={})"
        .format(blocksz, dpt, warp, blocksz // warp))
    out.append(render_waste_table(blocksz, dpt, demo_heights))
    out.append("")

    # Realistic geometry sweeps. shfl variants in SCAMP_VARIANT_TUPLES
    # default to BLOCKSZ=128 (SP) / 64 (DP) with a real warp of 32.
    real_blocksz_sp = 128
    real_blocksz_dp = 64
    real_warp = 32
    real_heights = sorted({dpt, 2 * dpt, 8 * dpt, 16 * dpt, real_warp * dpt,
                           2 * real_warp * dpt, 4 * real_warp * dpt})
    out.append(PREFIX + "REAL kernel SP geometry: BLOCKSZ={} DPT={} warp=32 "
               "(warps_per_block={})".format(
                   real_blocksz_sp, dpt, real_blocksz_sp // real_warp))
    out.append(render_waste_table(real_blocksz_sp, dpt, real_heights))
    out.append("")
    out.append(PREFIX + "REAL kernel DP geometry: BLOCKSZ={} DPT={} warp=32 "
               "(warps_per_block={})".format(
                   real_blocksz_dp, dpt, max(1, real_blocksz_dp // real_warp)))
    out.append(render_waste_table(real_blocksz_dp, dpt, real_heights))
    out.append("")

    out.append(PREFIX + "Closed form (valid when tile_height <= 32*DPT,")
    out.append(PREFIX + "i.e., the junk has room to propagate through the warp):")
    out.append(PREFIX + "  masked cells = T_h * (T_h - 1) / 2")
    out.append(PREFIX + "  waste %      = (T_h - 1) / (2 * BLOCKSZ * DPT) * T_h * 100")
    out.append(PREFIX + "    --> grows quadratically with tile_height, but")
    out.append(PREFIX + "        with BLOCKSZ in the denominator stays small")
    out.append(PREFIX + "        for the kernel's realistic blocksz=64/128.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate(blocksz, dpt, warp, rows, tile_height=None):
    assert blocksz % warp == 0, "BLOCKSZ must be a multiple of warp size"
    assert dpt >= 1
    if tile_height is None:
        tile_height = 32 * dpt  # one column-block rotation per tile

    out = []
    out.append(banner(
        "cov-shuffle: BLOCKSZ={} DPT={} WARP={} ROWS={}".format(
            blocksz, dpt, warp, rows)))
    out.append(
        "  design: each lane owns a FIXED DPT-wide column slice; cov\n"
        "  shuffles right one slot per row WITHIN a warp; lane 31 of\n"
        "  warp k publishes cov[DPT-1] to smem; lane 0 of warp k>0\n"
        "  receives that value. Lane 0 of warp 0 has no predecessor and\n"
        "  produces JUNK ('?') after row 0. Labels are block-local\n"
        "  diagonal indices.\n")

    width = compute_width(blocksz, dpt, warp, rows)
    state = initial_state(blocksz, dpt)

    for r in range(rows):
        out.append(banner("ROW {}".format(r)))
        mask = slot_validity(blocksz, dpt, r)
        out.append(render_state(state, mask, blocksz, dpt, warp, width,
                                label="BEFORE row {}: cov[i] tracks diag "
                                "(lane*DPT + i - row)".format(r)))
        out.append("")
        out.append(PREFIX + "(cov UPDATE in place: cov[i] += dfc[i]*dgr + "
                   "dgc[i]*dfr -- diagonals unchanged)")
        out.append("")

        new_state, handoff_writes, handoff_reads = step_one_row(
            state, blocksz, dpt, warp)

        if handoff_writes:
            out.append(render_handoff(handoff_writes, handoff_reads,
                                      blocksz, warp, width))
            out.append("")

        mask_after = slot_validity(blocksz, dpt, r + 1)
        out.append(render_state(new_state, mask_after, blocksz, dpt, warp,
                                width,
                                label="AFTER row {}: cov shuffled. (N) cells "
                                "are MASKED at next row.".format(r)))
        out.append("")
        state = new_state

    out.append(render_waste_summary(blocksz, dpt, warp, tile_height))
    out.append("")
    out.append(banner("legend", "-"))
    out.append("    L<T>     : lane T (block-local index)")
    out.append("    slot <i> : cov[i] of that lane")
    out.append("    N        : the cov slot is tracking block-local diagonal N")
    out.append("    ?        : junk -- no predecessor produced this value")
    out.append("    (N)      : the cov VALUE is fine, but this slot's diagonal")
    out.append("               is outside the block's meta-diag range; the")
    out.append("               distance computed here is masked from")
    out.append("               distc / distr updates.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocksz", type=int, default=8,
                   help="threads per block (must be multiple of --warp)")
    p.add_argument("--dpt", type=int, default=2,
                   help="diagonals per thread")
    p.add_argument("--warp", type=int, default=4,
                   help="warp size (use 4 or 8 for compact diagrams; "
                        "real hardware is 32)")
    p.add_argument("--rows", type=int, default=3,
                   help="number of rows of work to step through (visual frames)")
    p.add_argument("--tile-height", type=int, default=None,
                   help="tile_height for the waste-summary calc "
                        "(defaults to 32 * DPT, the kernel's max)")
    args = p.parse_args()

    if args.blocksz % args.warp != 0:
        print("error: --blocksz must be a multiple of --warp", file=sys.stderr)
        sys.exit(1)

    print(generate(args.blocksz, args.dpt, args.warp, args.rows,
                   args.tile_height))


if __name__ == "__main__":
    main()
