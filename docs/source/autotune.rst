GPU Autotuning
==============

SCAMP's GPU path picks one of several pre-built kernel variants (different
combinations of block size, diagonals-per-thread, tile height, etc.) for
each (profile type, precision) tuple at launch time. Different GPUs prefer
different variants — what wins on Ampere isn't necessarily what wins on
Pascal — so SCAMP carries an *autotune cache* that maps each device to
its preferred variant. This page covers what the cache is, where it
lives, and how to use it.

TL;DR
-----

* SCAMP ships with tuned configs for some GPUs, but it is not comprehensive.

  If you see a one-line warning at the top of a SCAMP run that starts
  with ``SCAMP: no autotune entry for device '<name>' ...``, your GPU
  isn't in the built-in cache. Run one of these once, then forget about
  it:

  .. code-block:: console

     # CLI:
     $ SCAMP --autotune

     # Python:
     >>> import pyscamp
     >>> pyscamp.autotune()

  The first SCAMP / pyscamp launch on a device without a cache entry
  prints a one-shot per-process warning recommending you run
  ``--autotune`` (subsequent launches in the same process are silent;
  see :ref:`autotune-miss-warning`). Silence it entirely with
  ``SCAMP_AUTOTUNE_QUIET=1``.

  This takes a few minutes to run and persists its choices to disk. The
  default location is ``~/.cache/scamp/autotune.txt`` on Linux/macOS and
  ``%LOCALAPPDATA%\scamp\autotune.txt`` on Windows. All subsequent SCAMP
  runs will pick up your tuned config automatically. See
  :ref:`autotune-default-path` for the full resolution rules.

How lookups work
----------------

When SCAMP launches a GPU kernel, it asks the autotuner for the best
config for the current ``(device, profile_type, precision)`` tuple.
The lookup tries these sources in order; the first hit wins:

1. **Per-thread override.** Used internally by the autotune benchmark
   loop to force a specific variant per timed trial. Not a user-facing
   knob.

2. **User cache** — ``~/.cache/scamp/autotune.txt`` on Linux/macOS or
   ``%LOCALAPPDATA%\scamp\autotune.txt`` on Windows by default (see
   :ref:`autotune-default-path` for the full resolution rules).
   Written by ``SCAMP --autotune`` / ``pyscamp.autotune()``. If you've
   tuned for your GPU, this is what gets used.

3. **Built-in cache.** ``data/autotune_cache.txt`` from the source
   tree, embedded into the binary at build time. This is how conda-forge
   and pip-wheel users get device-specific tuning without having to
   recompile — we ship entries for the GPUs we've benchmarked.

4. **Compile-time default.** A safe fallback variant. Works on every
   supported device but rarely the fastest. When SCAMP falls back to
   this you'll see a one-shot warning on stderr (see
   :ref:`autotune-miss-warning`).

.. _autotune-default-path:

Default cache location
----------------------

``--autotune`` writes to (and ``GetKernelConfigForDevice`` reads from)
the first of these paths that resolves:

1. ``$SCAMP_AUTOTUNE_CACHE`` — when set, used verbatim (Linux, macOS,
   and Windows).
2. ``$XDG_CACHE_HOME/scamp/autotune.txt`` — when ``XDG_CACHE_HOME`` is
   set (Linux default for users following the XDG Base Directory spec;
   honored on Windows too if explicitly set).
3. Platform-specific user dir:

   * Linux / macOS: ``$HOME/.cache/scamp/autotune.txt``.
   * Windows: ``%LOCALAPPDATA%\scamp\autotune.txt`` (typically
     ``C:\Users\<you>\AppData\Local\scamp\autotune.txt``), falling back
     to ``%USERPROFILE%\.cache\scamp\autotune.txt`` if ``LOCALAPPDATA``
     is unset.

The parent directory is created automatically by ``Save()`` if it
doesn't exist, so you don't need to ``mkdir -p`` it yourself.

Running the autotuner
---------------------

``SCAMP --autotune`` (or ``pyscamp.autotune()``) sweeps every enabled
variant × every supported block size for every ``(profile_type,
precision)`` pair and persists the per-tuple winner to the user cache.
A full sweep is the current variant count × 4 block sizes × 10 targets;
with the 5 variants enabled today that's 200 benchmark trials. With
the default benchmark workload (256K-element synthetic self-join) the
sweep takes ~10-20 minutes on a recent GPU; the output is verbose by
default so you can see progress.

Choosing the benchmark workload size
""""""""""""""""""""""""""""""""""""

The synthetic workload used per trial is sized via
``SCAMP_AUTOTUNE_INPUT_LENGTH`` (default 262144 = 256K elements).
Work scales like *n²*, so doubling the size roughly quadruples the
sweep's wall-clock cost — but the per-variant ranking gets tighter as
*n* grows: at small *n* the FFT/stats prelude dominates and trial
timings collapse toward the noise floor, while at production sizes the
kernel work swamps the prelude and the ranking is dominated by what
you actually care about.

Empirical comparison on an RTX 3080 across the standard 10
(profile, precision) targets:

============== ============== ============== ============== ===============
Input length   Sweep wall     Cross-target   Worst-case     Per-target
                              geomean        ratio          winners
============== ============== ============== ============== ===============
65536 (64K)    ~4 min         1.325          2.25           shift vs 128K
131072 (128K)  ~8 min         1.308          3.47           shift vs 256K
262144 (256K)  ~25 min        1.278          2.77           default
============== ============== ============== ============== ===============

The geomean ratio above is the cross-target "best recommended default"
score (lower is better — 1.000 would mean a single variant tied with
every per-target winner). The 256K row is meaningfully tighter than
128K (worst-case ratio drops 3.47 → 2.77, a ~20% reduction), and
importantly, the per-target *winners* themselves shift between the
rows (e.g. SUM_THRESH/DOUBLE picks different variants at 64K vs 128K
vs 256K), so a smaller-N autotune doesn't just mis-rank the
cross-target default — it picks suboptimal entries for individual
cache rows.

Run with a smaller value if 256K is impractically slow on your GPU
(older Pascal or T-class cards can take well over an hour at 256K),
or with a larger value when you're producing entries to ship in
``data/autotune_cache.txt``:

.. code-block:: console

   $ SCAMP_AUTOTUNE_INPUT_LENGTH=131072 SCAMP --autotune  # fast/casual
   $ SCAMP_AUTOTUNE_INPUT_LENGTH=524288 SCAMP --autotune  # tighter still

The trade-off is wall-clock: 256k takes 16x longer than 128k.

Another important note is that if you don't plan on running large joins in
practice, tuning with a smaller input size is more relevant to your workload.
If you will only use SCAMP on smaller inputs there is no need to tune for a
larger size.

Choosing the device(s)
""""""""""""""""""""""

Both the CLI and ``pyscamp.autotune()`` default to **device 0 only** —
on a multi-GPU box with identical devices, tuning them all wastes time.
Override explicitly if you really do need to tune a second physical GPU
type:

.. code-block:: console

   $ SCAMP --autotune --gpus=0,1            # CLI
   >>> pyscamp.autotune(devices=[0, 1])     # Python

.. _autotune-miss-warning:

The "no autotune entry" warning
-------------------------------

The first time SCAMP can't find an autotune entry for a given
``(device, profile, precision)`` tuple in any cache source, it emits a
one-shot warning to stderr that looks like:

.. code-block:: text

   SCAMP: no autotune entry for device 'NVIDIA_T1000__sm_75' / 1NN_INDEX / SINGLE; using compile-time default (blocksz=128 bps=8 dpt=4 ur=0 our=8 kti=8).
     Run `SCAMP --autotune` or `pyscamp.autotune()` to benchmark a better config for this device.
     (Suppressing further warnings for this tuple.)

This is informational, not an error — SCAMP will run correctly with the
default config, just not as fast as it could be. Two follow-ups:

* Run ``--autotune`` once to populate your local cache. This silences
  the warning for that tuple on subsequent runs and gives you a
  measurable speed-up.

* Optionally open a PR adding your device's lines from the user cache
  (see :ref:`autotune-default-path` for its location on your platform)
  into ``data/autotune_cache.txt`` so the next release ships those
  entries to other users of your GPU.

Silencing the warning
"""""""""""""""""""""

The warning prints to stderr by default for both the CLI and pyscamp.
Silence it via the ``SCAMP_AUTOTUNE_QUIET`` environment variable:

.. code-block:: console

   # Force-silence the warning (e.g. in a CI run or notebook session):
   $ SCAMP_AUTOTUNE_QUIET=1 SCAMP ...
   $ SCAMP_AUTOTUNE_QUIET=1 python -c "import pyscamp; pyscamp.selfjoin(...)"

Other autotune environment variables
------------------------------------

A handful of additional env vars let you tune SCAMP's autotune /
launch-time behavior without rebuilding. All are read on first use and
their value is cached for the lifetime of the process — re-export
changes after first use have no effect.

``SCAMP_AUTOTUNE_PRECISION_FILTER``
    ``SINGLE`` | ``DOUBLE`` | ``all`` (default). Restricts ``SCAMP
    --autotune`` (and ``pyscamp.autotune()``) to one precision.
    Filtered targets are reported as ``SKIPPED`` and their cache entries
    are left untouched, so you can re-run for the other precision
    without losing the existing entries.

``SCAMP_AUTOTUNE_VARIANT_FILTER``
    ``shfl`` | ``sliding-window`` (also ``sw`` / ``smem``) | ``all``
    (default). Restricts the autotune sweep to one variant family.
    ``shfl`` matches every variant with ``unrolled_rows == 0`` (the
    cov-shuffle kernel); the other names match the sliding-window
    kernel. Useful when iterating on a specific kernel family without
    sweeping the other.

``SCAMP_AUTOTUNE_WARMUP_RUNS``
    Per-trial warmup count for the autotuner's bench function. Default
    ``0``: the first launch of a given ``(variant, blocksz)``
    instantiation is typically only a few percent slower than
    steady-state because most JIT / module-load cost is amortized by
    the process-level first launch, and the cross-target geomean
    ranking tolerates a few percent of noise. Set to ``1`` (or more)
    when trial timings look noisy or on a colder GPU/driver where the
    first launch of a never-before-seen template instantiation takes
    significantly longer than steady-state. Value is cached at first
    autotune call.

``SCAMP_FORCE_VARIANT``
    Index of a single GPU kernel variant to force for every
    ``(profile, precision)`` launch, bypassing the autotune cache and
    cold-start default. The precision still picks the cold-start
    blocksz; the full ``{64, 128, 256, 512}`` blocksz axis is NOT
    swept here. Used by CI to exercise each compiled variant against
    the correctness test suite without writing per-variant cache
    files. Valid indices are reported by ``SCAMP --list_variants``.
    Out-of-range or malformed values silently fall through to the
    normal lookup path. Value is cached at first kernel launch.

The value is interpreted as truthy unless it's exactly ``0``, ``false``,
``FALSE``, or empty.

Clearing or resetting the cache
-------------------------------

The user cache is a plain-text file at the location described in
:ref:`autotune-default-path`. To start fresh:

.. code-block:: console

   # Linux / macOS:
   $ rm ~/.cache/scamp/autotune.txt

   # Windows (PowerShell):
   > Remove-Item "$env:LOCALAPPDATA\scamp\autotune.txt"

The next SCAMP run will fall through to the built-in cache (and emit a
miss warning if your device isn't shipped). Running ``--autotune``
again regenerates the file.

If you suspect the user cache has a bad entry but don't want to delete
the file (e.g. it has good entries for *some* devices), you can edit it
by hand — each line is one record, ``#`` starts a comment, and the
format is documented in the file's own header.

To bypass the user cache entirely without deleting it, point
``SCAMP_AUTOTUNE_CACHE`` at an empty file:

.. code-block:: console

   $ touch /tmp/empty_cache.txt
   $ SCAMP_AUTOTUNE_CACHE=/tmp/empty_cache.txt SCAMP ...

The built-in cache still applies; only the user override is bypassed.

What happens to my cache when I upgrade SCAMP?
----------------------------------------------

By default, an existing user cache survives the upgrade — the file
format is keyed by the variant geometry tuple
``(blocks_per_sm, diags_per_thread, unrolled_rows, outer_unrolled_rows,
kernel_tile_iters)``, not by a position in some table, so cache entries
that still match a current kernel variant continue to hit.

The three things that can happen to an existing entry after an upgrade:

* **The new SCAMP build still has your entry's variant tuple.** Lookup
  succeeds and you keep your tuned config. This is the common case
  when a release just adds new variants.
* **The new build retired your entry's variant.** The runtime rejects
  the entry (it doesn't match any current variant), falls through to
  the built-in cache, and — if that also misses — to the compile-time
  default plus a one-shot "no autotune entry" warning. Other entries
  in the same cache file are unaffected; only the one(s) naming the
  retired variant fall through. You can ignore the warning, or run
  ``--autotune`` again to refresh the affected ``(device, profile,
  precision)`` tuples.
* **The release bumped the cache file's version header.** This is
  reserved for hard-incompatibility changes (the file schema changed,
  or kernel semantics shifted enough that *every* tuned config is
  stale). The new SCAMP silently treats the file as empty —
  ``--autotune`` is required to get back to a tuned state. SCAMP will
  not throw or refuse to run; you'll just see the cache-miss warning
  until you re-tune.

You don't need to delete your cache after a SCAMP upgrade. Run
``--autotune`` again if a release note tells you to (or if you see
the cache-miss warning after an upgrade and want to make it go away);
otherwise, your existing entries keep being used wherever they remain
applicable.

Troubleshooting
---------------

**"I updated** ``data/autotune_cache.txt`` **but the binary still uses the
old values."**
   CMake re-runs ``configure_file`` when the cache file changes (via
   ``CMAKE_CONFIGURE_DEPENDS``), so a plain ``cmake --build .`` is enough
   to pick up the new contents. If you suspect a stale embed, verify
   with ``strings build/SCAMP | grep <your_device_key>``; the embedded
   string should match the on-disk file.

**"My configs aren't being respected on a multi-GPU box."**
   The cache is keyed by sanitized device name + compute capability
   (e.g. ``NVIDIA_GeForce_RTX_3080__sm_86``). If you have two different
   GPU models, you need entries for both — autotune device 0 first,
   then re-run with ``--gpus=1`` (or ``devices=[1]``) for the second.

**"I want to test a specific variant by hand."**
   Edit the user cache directly (``~/.cache/scamp/autotune.txt`` on
   Linux/macOS, ``%LOCALAPPDATA%\scamp\autotune.txt`` on Windows): each
   line is ``device_key|profile|precision|blocksz|bps|dpt|ur|our|kti``.
   However, you must specify a valid variant defined in 
   ``src/core/gpu_kernel/CMakeLists.txt``; the variants have to be
   specified at BUILD time for them to be included in the SCAMP binary.
   Lines that name an unknown variant are silently rejected at lookup
   time and the next source is consulted, so it's safe to experiment.
