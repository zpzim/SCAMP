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

* Most users do nothing. SCAMP ships with tuned configs for common GPUs
  baked into the binary; the autotuner only needs to run when your GPU
  isn't in that built-in list.

* If you see a one-line warning at the top of a SCAMP run that starts
  with ``SCAMP: no autotune entry for device '<name>' ...``, your GPU
  isn't in the built-in cache. Run one of these once, then forget about
  it:

  .. code-block:: console

     # CLI:
     $ SCAMP --autotune

     # Python:
     >>> import pyscamp
     >>> pyscamp.autotune()

  This takes a few minutes to run and persists its choices to disk
  (``~/.cache/scamp/autotune.txt`` on Linux by default). All subsequent
  SCAMP runs will pick up your tuned config automatically.

How lookups work
----------------

When SCAMP launches a GPU kernel, it asks the autotuner for the best
config for the current ``(device, profile_type, precision)`` tuple.
The lookup tries these sources in order; the first hit wins:

1. **Per-thread override.** Used internally by the autotune benchmark
   loop to force a specific variant per timed trial. Not a user-facing
   knob.

2. **User cache** — ``~/.cache/scamp/autotune.txt`` by default
   (see :ref:`autotune-default-path` for the full resolution rules).
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

1. ``$SCAMP_AUTOTUNE_CACHE`` — when set, used verbatim.
2. ``$XDG_CACHE_HOME/scamp/autotune.txt`` — when ``XDG_CACHE_HOME`` is
   set (Linux default for users following the XDG Base Directory spec).
3. ``$HOME/.cache/scamp/autotune.txt`` — on most Linux/Mac setups this
   is where you'll end up.

The parent directory is created automatically by ``Save()`` if it
doesn't exist, so you don't need to ``mkdir -p`` it yourself.

Running the autotuner
---------------------

``SCAMP --autotune`` (or ``pyscamp.autotune()``) sweeps every enabled
variant × every supported block size for every ``(profile_type,
precision)`` pair and persists the per-tuple winner to the user cache.
A full sweep is roughly 9 variants × 4 block sizes × 10 targets = 360
benchmark trials. With the default benchmark workload (131K-element
synthetic self-join) the sweep takes ~3-5 minutes on a recent GPU; the
output is verbose by default so you can see progress.

Choosing the benchmark workload size
""""""""""""""""""""""""""""""""""""

The synthetic workload used per trial is sized via
``SCAMP_AUTOTUNE_INPUT_LENGTH`` (default 131072). Larger values are
slower (work scales like *n²*) but the per-variant ranking better
matches a production-scale workload — at small *n* the FFT/stats prelude
dominates and trial timings collapse into the noise floor. For
production-quality tuning we recommend 256K-512K:

.. code-block:: console

   $ SCAMP_AUTOTUNE_INPUT_LENGTH=524288 SCAMP --autotune

The trade-off is wall-clock: a 524288 sweep can take 10-15 minutes.

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

* Optionally open a PR adding your device's lines from
  ``~/.cache/scamp/autotune.txt`` into ``data/autotune_cache.txt`` so
  the next release ships those entries to other users of your GPU.

Silencing the warning
"""""""""""""""""""""

The warning prints to stderr by default for CLI users (where it shows up
plainly on the terminal) and is silenced by default for pyscamp users
(where stray stderr output in a notebook is unwelcome — pyscamp's module
init sets ``SCAMP_AUTOTUNE_QUIET=1`` if it isn't already set).

You can override the default in either direction with the
``SCAMP_AUTOTUNE_QUIET`` environment variable:

.. code-block:: console

   # Force-silence the warning (e.g. in a CI run):
   $ SCAMP_AUTOTUNE_QUIET=1 SCAMP ...

   # Force-enable it under pyscamp (set the env var before importing):
   $ SCAMP_AUTOTUNE_QUIET=0 python -c "import pyscamp; pyscamp.selfjoin(...)"

The value is interpreted as truthy unless it's exactly ``0``, ``false``,
``FALSE``, or empty.

Clearing or resetting the cache
-------------------------------

The user cache is a plain-text file at the location described in
:ref:`autotune-default-path`. To start fresh:

.. code-block:: console

   $ rm ~/.cache/scamp/autotune.txt

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

Troubleshooting
---------------

**"I updated** ``data/autotune_cache.txt`` **but the binary still uses the
old values."**
   CMake re-runs ``configure_file`` when the cache file changes (via
   ``CMAKE_CONFIGURE_DEPENDS``), so a plain ``cmake --build .`` is enough
   to pick up the new contents. If you suspect a stale embed, verify
   with ``strings build/SCAMP | grep <your_device_key>``; the embedded
   string should match the on-disk file.

**"** ``pyscamp.autotune()`` **finishes in milliseconds without printing any
trials."**
   You're on a version older than the fix that switched the Python
   binding from the ``RunAutotune`` stub to ``RunAutotuneWithBenchmark``.
   Upgrade pyscamp (or run from a recent main branch).

**"My configs aren't being respected on a multi-GPU box."**
   The cache is keyed by sanitized device name + compute capability
   (e.g. ``NVIDIA_GeForce_RTX_3080__sm_86``). If you have two different
   GPU models, you need entries for both — autotune device 0 first,
   then re-run with ``--gpus=1`` (or ``devices=[1]``) for the second.

**"I want to test a specific variant by hand."**
   Edit ``~/.cache/scamp/autotune.txt`` directly: each line is
   ``device_key|profile|precision|blocksz|bps|dpt|ur|our|kti``. Lines
   that name an unknown variant are silently rejected at lookup time
   and the next source is consulted, so it's safe to experiment.
