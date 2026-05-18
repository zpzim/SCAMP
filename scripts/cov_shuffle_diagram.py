#!/usr/bin/env python3
"""Generate ASCII diagrams of the cov-shuffle algorithm used by variant 6
(and the per-warp-independent variant 8 follow-up).

Each frame shows, for one row of work inside a tile:
  - the BEFORE state: which diagonal each lane's cov[i] slot is currently
    tracking;
  - the cov UPDATE (in-place, advances cov along its diagonal — slot
    identity unchanged);
  - the SHUFFLE (cross-warp via smem hand-off for variant 6 / intra-warp
    wrap-around for variant 8) + the within-lane right-shift;
  - the AFTER state.

The "diagonal" labels are block-local diagonal indices (column - row in
block-local coordinates). The block has lanes 0..BLOCKSZ-1; each lane T
owns columns [T*DPT, T*DPT + DPT - 1]. At row r, lane T's cov[i]
TRACKS diagonal (T*DPT + i - r). A slot is INVALID (gets masked from
distc/distr) when its diagonal is outside the block's responsibility:

  - variant 6 (cross-warp): lane T cov[i] invalid iff (T*DPT + i - r) < 0
    (i.e., diagonal is to the left of the block's meta-diagonal range).
    Only lane 0 of warp 0 ever produces invalid slots within a tile.

  - variant 8 (per-warp-independent): each warp k owns its own 32*DPT-
    diagonal sub-range [k*32*DPT, (k+1)*32*DPT). cov in lane T cov[i]
    at row r tracks "warp-local diagonal" (T*DPT + i - r) mod (32*DPT)
    because the intra-warp shuffle wraps lane 0 <- lane 31. The slot is
    INVALID for the current warp iff the row-shifted diagonal has
    wrapped past 0 (i.e., r > T*DPT + i).

Run as:
    python3 cov_shuffle_diagram.py [--blocksz N] [--dpt N] [--warp N]
                                   [--rows N] [--variant 6|8]

Defaults are tiny (BLOCKSZ=8, DPT=2, WARP=4, ROWS=3) so the diagrams
fit on one screen. Realistic configs (BLOCKSZ=256, WARP=32) generate
multi-page output; pipe to less.
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

JUNK = "?"  # value held by a slot whose tracked diagonal is invalid


def initial_state(blocksz, dpt):
    """Return list-of-lists: state[lane][slot] = diagonal-label string.

    At row 0, lane T cov[i] = cov(0, T*DPT + i) on diagonal (T*DPT + i).
    We label slots with their global block-local diagonal index.
    """
    return [[str(t * dpt + i) for i in range(dpt)] for t in range(blocksz)]


def step_one_row(state, blocksz, dpt, warp, variant):
    """Advance the cov state by one row.

    Returns (new_state, handoff_writes, handoff_reads) where:
      handoff_writes[warpid] = label written by lane 31 of warp warpid
      handoff_reads[warpid] = label read by lane 0 of warp warpid (or None)

    Both lists are empty for variant 8.
    """
    new_state = [[None] * dpt for _ in range(blocksz)]
    handoff_writes = {}
    handoff_reads = {}

    warps_per_block = blocksz // warp

    # Compute hand-off values BEFORE updating any lane. For variant 6, each
    # warp's lane 31 publishes its current cov[DPT-1] for the NEXT warp's
    # lane 0 to consume.
    if variant == 6:
        for wid in range(warps_per_block):
            lane31 = wid * warp + (warp - 1)
            handoff_writes[wid] = state[lane31][dpt - 1]
        # The "read" by lane 0 of warp wid is what lane 31 of warp wid-1
        # WROTE LAST ROW. For the diagram we display this row's read =
        # last row's write. But since this function only knows THIS row,
        # we compute "read = previous warp's published value as of this
        # row" -- i.e., the same content lane 31 holds NOW (which IS
        # what it would have published last row, since cov accumulates).
        # In practice the value differs by one row of accumulation, but
        # for the diagonal-label tracking that we care about, it is the
        # same diagonal index.
        for wid in range(1, warps_per_block):
            handoff_reads[wid] = handoff_writes[wid - 1]

    # Per-lane intra-warp shuffle: shift cov[i] right within the lane;
    # cov[0] receives shuffled-in value.
    for warp_id in range(warps_per_block):
        for lane_in_warp in range(warp):
            T = warp_id * warp + lane_in_warp
            # Shift right: new cov[i] = old cov[i-1] for i >= 1.
            for i in range(dpt - 1, 0, -1):
                new_state[T][i] = state[T][i - 1]
            # cov[0]: predecessor lane's cov[DPT-1]. Predecessor depends
            # on variant.
            if lane_in_warp > 0:
                # Intra-warp predecessor (both variants).
                pred = warp_id * warp + (lane_in_warp - 1)
                new_state[T][0] = state[pred][dpt - 1]
            else:
                # Lane 0 of a warp.
                if variant == 6:
                    if warp_id > 0:
                        # Cross-warp hand-off (received from prev warp's
                        # smem-published value).
                        new_state[T][0] = handoff_reads[warp_id]
                    else:
                        # Lane 0 of warp 0: no predecessor warp.
                        new_state[T][0] = JUNK
                else:  # variant 8
                    # Per-warp-independent: wrap inside the warp from
                    # lane 31's cov[DPT-1]. The value is "valid" as a cov
                    # for some diagonal but for the WRONG diagonal for
                    # lane 0's column, hence masked downstream.
                    lane31_of_warp = warp_id * warp + (warp - 1)
                    new_state[T][0] = state[lane31_of_warp][dpt - 1]

    return new_state, handoff_writes, handoff_reads


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------

def slot_validity(blocksz, dpt, warp, row, variant):
    """Return mask[lane][slot] in {'.', 'X'} where 'X' means INVALID
    (will be masked from distc/distr).

    Variant 6: only lane 0 of warp 0 produces invalid slots; the slot
    becomes invalid when its tracked diagonal goes negative.

    Variant 8: each warp's lane 0 produces invalid slots, and the
    invalidation walks down through the warp at one slot per row.
    """
    mask = [['.'] * dpt for _ in range(blocksz)]
    warps_per_block = blocksz // warp
    for warp_id in range(warps_per_block):
        for lane_in_warp in range(warp):
            T = warp_id * warp + lane_in_warp
            local_col = T * dpt
            for i in range(dpt):
                if variant == 6:
                    # cell (r, local_col + i) is in block's meta-diag range
                    # iff local_col + i >= row.
                    if local_col + i < row:
                        mask[T][i] = 'X'
                else:  # variant 8
                    # Each warp owns 32*DPT diagonals. The slot at lane T
                    # in warp k slot i becomes invalid when r > T*DPT + i
                    # in WARP-LOCAL terms (i.e., relative to the warp).
                    warp_local_T = lane_in_warp
                    if warp_local_T * dpt + i < row:
                        mask[T][i] = 'X'
    return mask


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

def cell_width(state):
    """Pick a column width wide enough for any cell label."""
    longest = 1
    for row in state:
        for v in row:
            longest = max(longest, len(str(v)))
    return max(2, longest)


def render_state(state, mask, blocksz, dpt, warp, label=None, width=None):
    """Render the block as a grid: rows = warps, columns = lanes.

    Each lane is a column of DPT cells (slot 0 on top, slot DPT-1 on bottom).
    Warps are separated by a horizontal rule. Each cell shows the diagonal
    label; if the slot is INVALID per `mask`, the label is parenthesized.
    """
    if width is None:
        width = cell_width(state)

    warps_per_block = blocksz // warp
    out = []
    if label:
        out.append(label)

    # Header: lane indices
    for warp_id in range(warps_per_block):
        # Lane index header (block-local)
        lane_idx_row = "  warp " + str(warp_id).rjust(2) + " :"
        for lane_in_warp in range(warp):
            T = warp_id * warp + lane_in_warp
            lane_idx_row += " " + ("L" + str(T)).rjust(width)
        out.append(lane_idx_row)
        # DPT rows: one per slot
        for slot in range(dpt):
            slot_row = "  slot " + str(slot) + "    :"
            for lane_in_warp in range(warp):
                T = warp_id * warp + lane_in_warp
                v = str(state[T][slot])
                if mask[T][slot] == 'X':
                    v = "(" + v + ")"
                slot_row += " " + v.rjust(width)
            out.append(slot_row)
        # Spacer between warps
        if warp_id < warps_per_block - 1:
            out.append("  " + "-" * (10 + (width + 1) * warp))
    return "\n".join(out)


def render_handoff(handoff_writes, handoff_reads, blocksz, warp, width):
    """Render the smem cov_handoff region usage for variant 6.

    Shows: for each warp, what its lane 31 WROTE to smem, and what its
    lane 0 READ from smem (= previous warp's lane 31's write).
    """
    warps_per_block = blocksz // warp
    out = ["  smem cov_handoff[warps_per_block=" + str(warps_per_block) + "]:"]
    header = "    warp:    "
    for wid in range(warps_per_block):
        header += " " + ("w" + str(wid)).rjust(width)
    out.append(header)
    write_row = "    lane31 W:"
    for wid in range(warps_per_block):
        v = str(handoff_writes.get(wid, "-"))
        write_row += " " + v.rjust(width)
    out.append(write_row)
    read_row = "    lane0  R:"
    for wid in range(warps_per_block):
        v = handoff_reads.get(wid, None)
        s = "-" if v is None else str(v)
        read_row += " " + s.rjust(width)
    out.append(read_row)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def banner(text, char="="):
    bar = char * max(60, len(text) + 4)
    return "\n".join([bar, "  " + text, bar])


def generate(blocksz, dpt, warp, rows, variant):
    assert blocksz % warp == 0, "BLOCKSZ must be a multiple of warp size"
    assert dpt >= 1
    out = []
    out.append(banner(
        "variant {} cov-shuffle: BLOCKSZ={} DPT={} WARP={} rows={}".format(
            variant, blocksz, dpt, warp, rows)))
    if variant == 6:
        out.append(
            "design: each lane owns a FIXED DPT-wide column slice; cov\n"
            "        shuffles right one slot per row WITHIN a warp; lane 31\n"
            "        of warp k publishes cov[DPT-1] to smem; lane 0 of warp\n"
            "        k > 0 receives that value. Lane 0 of warp 0 has no\n"
            "        predecessor and produces JUNK ('?') after row 0.\n"
            "        Labels are block-local diagonal indices.\n")
    elif variant == 8:
        out.append(
            "design: per-warp-independent. Each warp's cov shuffle WRAPS\n"
            "        within the warp (lane 0 <- lane 31). NO smem hand-off\n"
            "        and NO __syncthreads(). The wrapped values are valid\n"
            "        covs for OTHER diagonals -- the slot validity mask\n"
            "        catches them. Every warp's lane 0 produces invalid\n"
            "        slots after some row.\n")

    state = initial_state(blocksz, dpt)
    width = cell_width(state)

    for r in range(rows):
        out.append(banner("ROW {}".format(r)))
        # Before
        mask = slot_validity(blocksz, dpt, warp, r, variant)
        out.append(render_state(state, mask, blocksz, dpt, warp,
                                label="  BEFORE row {}: cov[i] tracks "
                                "diag (lane*DPT + i - row)".format(r),
                                width=width))
        # Cov update is in-place and doesn't change the diagonal each
        # slot tracks, so we don't render a separate "after update" frame.
        out.append("\n  (cov UPDATE in place: cov[i] += dfc[i]*dgr + "
                   "dgc[i]*dfr -- diagonals unchanged)\n")

        # Step
        new_state, handoff_writes, handoff_reads = step_one_row(
            state, blocksz, dpt, warp, variant)

        if variant == 6 and handoff_writes:
            out.append(render_handoff(handoff_writes, handoff_reads,
                                      blocksz, warp, width))
            out.append("")

        # After
        mask_after = slot_validity(blocksz, dpt, warp, r + 1, variant)
        out.append(render_state(new_state, mask_after, blocksz, dpt, warp,
                                label="  AFTER row {}: cov shuffled. Cells "
                                "in (parens) are MASKED next row.".format(r),
                                width=width))
        out.append("")
        state = new_state

    out.append(banner("legend", "-"))
    out.append("  L<T>     : lane T (block-local index)")
    out.append("  slot <i> : cov[i] of that lane")
    out.append("  N        : the cov slot is tracking block-local diagonal N")
    out.append("  ?        : junk -- no predecessor produced this value")
    out.append("  (N)      : valid cov for diagonal N, but for this lane's")
    out.append("             column position the diagonal is outside the")
    out.append("             block's meta-diag range; the distc/distr update")
    out.append("             is masked.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocksz", type=int, default=8,
                   help="threads per block (must be multiple of --warp)")
    p.add_argument("--dpt", type=int, default=2,
                   help="diagonals per thread")
    p.add_argument("--warp", type=int, default=4,
                   help="warp size (use 4 or 8 for compact diagrams; "
                        "real hardware is 32)")
    p.add_argument("--rows", type=int, default=3,
                   help="number of rows of work to step through")
    p.add_argument("--variant", type=int, default=6, choices=[6, 8],
                   help="6 = cross-warp via smem; 8 = per-warp-independent")
    args = p.parse_args()

    if args.blocksz % args.warp != 0:
        print("error: --blocksz must be a multiple of --warp", file=sys.stderr)
        sys.exit(1)

    print(generate(args.blocksz, args.dpt, args.warp, args.rows, args.variant))


if __name__ == "__main__":
    main()
