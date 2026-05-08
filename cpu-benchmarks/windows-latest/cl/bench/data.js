window.BENCHMARK_DATA = {
  "lastUpdate": 1778252992929,
  "repoUrl": "https://github.com/zpzim/SCAMP",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "Zach Zimmerman",
            "username": "zpzim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a16f33c498807b56e0105d5fc23344aaad676015",
          "message": "Improve benchmarking to add stable CPU benchmarks and reduce variance on Github runners. (#118)\n\n* Add a stable version of gcc/clang benchmarks on self-hosted linux box.\r\n\r\n* Update benchmarking to run smaller benchmarks. Output timing information in seconds.\r\n\r\n* Update docs to point to stable benchmark suites.",
          "timestamp": "2022-06-18T10:52:55-07:00",
          "tree_id": "aa3c204e0fb719b7016d86a95d7232bca453e9a2",
          "url": "https://github.com/zpzim/SCAMP/commit/a16f33c498807b56e0105d5fc23344aaad676015"
        },
        "date": 1655575228033,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.5043976999999813,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1.1754224499999963,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.4530951599999753,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.00625 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.4511537000003045,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.015625 s\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "Zach Zimmerman",
            "username": "zpzim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ba40129d7615c06a2cc186b720e25183f4b5c20a",
          "message": "Add GPU integration tests. (#81)\n\nAdds GPU integration tests to verify output correctness of GPU kernels.",
          "timestamp": "2022-06-18T13:37:28-07:00",
          "tree_id": "4f69ab291f6b937e5b36b8aae3bb2c4ae203ed4f",
          "url": "https://github.com/zpzim/SCAMP/commit/ba40129d7615c06a2cc186b720e25183f4b5c20a"
        },
        "date": 1655586297782,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.528200030000005,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0046875 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1.20586626999999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0015625 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.4724289100000079,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 4.248414000000025,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0 s\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "Zach Zimmerman",
            "username": "zpzim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "27e617febc69c476408ae05b217be395cc72aa35",
          "message": "Fix some broken links in intro.rst (#119)",
          "timestamp": "2022-08-04T07:51:56-07:00",
          "tree_id": "dd44769e8b2614928cd8d09a124fe81845fdc641",
          "url": "https://github.com/zpzim/SCAMP/commit/27e617febc69c476408ae05b217be395cc72aa35"
        },
        "date": 1659625527012,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.484909719999996,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1.2605377000000089,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0015625 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.5153671300000042,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.9791477999999643,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0 s\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "Zach Zimmerman",
            "username": "zpzim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fe867f9cf755d149f1a7aa98b6ed0509c1311fac",
          "message": "Update README.md to include DIO from zenodo",
          "timestamp": "2023-07-31T17:14:19-07:00",
          "tree_id": "d2988c8498c94c5b3c4019ae370c118301eb5783",
          "url": "https://github.com/zpzim/SCAMP/commit/fe867f9cf755d149f1a7aa98b6ed0509c1311fac"
        },
        "date": 1690849548237,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.7357924600000048,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1.4593165100000078,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0046875 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.80053165999999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 4.414454599999999,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0 s\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "Zach Zimmerman",
            "username": "zpzim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3f3078e9abdfd4141fe19ce68500d9ab908353ba",
          "message": "Add support for cuda 12 builds (#124)\n\n* Adds support for builing for compute capabilities 87, 89, and 90. \r\n\r\n* Fixes issues with CUDA_ARCHITECTURES not being set correctly based on cuda compiler version.\r\n\r\n* Fix some broken test scripts.\r\n\r\n* Bump SDE version for arch emulation test",
          "timestamp": "2024-01-08T00:35:29-08:00",
          "tree_id": "4bf8750b26a17e93ab412acda21fe81cf34f943c",
          "url": "https://github.com/zpzim/SCAMP/commit/3f3078e9abdfd4141fe19ce68500d9ab908353ba"
        },
        "date": 1704703867550,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.9664529299999913,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0046875 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.7188500900000008,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.951027269999986,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.923651899999868,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.015625 s\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "Zach Zimmerman",
            "username": "zpzim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1b5a3b19cd05c208c3b01982f691c4a26ee4dbe5",
          "message": "Update docker build to cuda 12.3.1 (#125)\n\n* Update Docker image to use CUDA 12.3.1\r\n\r\n* Update grpc submodule to v1.60.0\r\n\r\n* Fix client/server build issue with new grpc.",
          "timestamp": "2024-01-08T20:12:11-08:00",
          "tree_id": "5c1a872dab9181e06e3382673077c62069ee7d9b",
          "url": "https://github.com/zpzim/SCAMP/commit/1b5a3b19cd05c208c3b01982f691c4a26ee4dbe5"
        },
        "date": 1704774705247,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.974676199999999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0015625 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.7233457999999928,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.9525226399999951,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.9312444000000824,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0 s\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "Zach Zimmerman",
            "username": "zpzim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "aaab3a2d072706333fa3f89193af337108d22d4f",
          "message": "Modernize the build for Python 3.13/3.14, NumPy 2, CUDA 12.x, and CMake 4 (#136)\n\n* setup.py: drop unused distutils.version import\n\ndistutils was removed from the standard library in Python 3.12 and\nthe setuptools shim was removed in setuptools 74. The LooseVersion\nimport here was never actually used anywhere in setup.py, so the\nfix is to delete the line.\n\nRefs: https://docs.python.org/3.12/whatsnew/3.12.html#distutils\n\n* Bump vendored pybind11 from v2.9.2 to v2.13.6\n\nThe vendored pybind11 was pinned at v2.9.2 (March 2022), which\npredates stable support for:\n  - Python 3.12 free-threading prep\n  - Python 3.13 stable ABI       (needs >= 2.12)\n  - Python 3.14                  (needs >= 2.13.6)\n  - NumPy 2 dtype handling fixes (numerous bugfixes 2.10 -> 2.13)\n\nv2.13.6 is the last point release before pybind11 3.x and is the\nbroadest-compatibility option, supporting CPython 3.7 through 3.13\nwith experimental 3.14 support.\n\nAlso bump the floor in src/python/CMakeLists.txt's find_package call\nso users opting into PYSCAMP_USE_EXTERNAL_PYBIND11 get the same\nguarantees as the vendored build.\n\n* Fix numpy>=2 selfjoin returning a constant vector (#129)\n\nRoot cause: pybind11's automatic conversion of a numpy ndarray to\n`std::vector<double>` (via pybind11/stl.h's list_caster) iterates the\narray as a Python sequence and round-trips every element through\nPython objects.  Under NumPy >= 2 with the older vendored pybind11\n(v2.9.2), this iteration path silently produced zeroed inputs for\nsome 1D float64 arrays.  SCAMP then ran on a degenerate (zero-variance)\ntime series, every Pearson correlation came out as the 'no match'\nsentinel, and CleanupPearson() turned every output into NaN -- the\nconstant-output symptom users reported.\n\nFix: stop relying on stl.h's caster.  Each binding now takes\n`py::array_t<double, py::array::c_style | py::array::forcecast>`,\nwhich goes through the buffer protocol with an explicit dtype\nrequest.  We then do one bulk memcpy into a std::vector<double>\ninside the new ArrayToDoubleVector() helper.  This is:\n\n  - dtype-safe: any 1D numeric array (float32/int64/etc.) is silently\n    upcast to float64 with a contiguous copy if needed,\n  - faster: a single contiguous copy instead of N Python-object round\n    trips,\n  - identical behavior on NumPy 1.x and 2.x.\n\nThe underlying scamp/scamp_sum/scamp_knn/scamp_matrix C++ functions\nkeep their existing `const std::vector<double>&` signatures so the\ninternal call sites and the function-pointer overload resolution\ntable are unchanged.\n\nCloses #129\n\n* Add SCAMP_DISABLE_THRUST_SORT escape hatch for buggy CCCL toolchains\n\nBackground: with the CUDA 12.9 + GCC 13 toolchain shipped by current\nconda-forge (and likely other distributions), including <thrust/sort.h>\nfails at compile time with:\n\n    thrust/detail/use_default.h:29:1: error: macro\n    \"_CCCL_PP_SPLICE_WITH_IMPL1\" passed 3 arguments, but takes just 2\n    THRUST_NAMESPACE_BEGIN\n\nThis is a CCCL preprocessor regression (the same _CCCL_PP_SPLICE_WITH_IMPL1\nhelper is defined as both a 2-arg and 3-arg macro depending on which CCCL\nheaders reach the translation unit first).  It is upstream's bug, not\nSCAMP's, and is a moving target across CUDA point releases.\n\nSCAMP only uses thrust in one place: `match_gpu_sort` for KNN match\nordering.  This commit adds a CMake option `SCAMP_DISABLE_THRUST_SORT`\n(default OFF, behavior unchanged) that, when ON:\n\n  - removes the <thrust/...> includes from kernels.cu entirely, so the\n    CCCL macro path is never instantiated, and\n  - replaces `thrust::sort` with a host-side std::sort fallback that\n    round-trips matches through host memory.\n\nThe KNN match arrays are typically small enough that the host round-trip\nisn't the bottleneck.  If profiling ever shows otherwise, the natural\nfollow-up is to use cub::DeviceMergeSort::SortKeys instead.\n\nDownstream packagers hitting the CCCL issue (e.g. the conda-forge\npyscamp-gpu-feedstock CUDA 12.9 PR) can pass\n-DSCAMP_DISABLE_THRUST_SORT=ON via PYSCAMP_ADD_CMAKE_ARGS to unblock\nthe build without waiting for an upstream CCCL fix.\n\n* ci: cover Python 3.13 + NumPy 2 and CUDA 12.6 + thrust-sort opt-out\n\nAdds two pieces of regression coverage that the existing CI matrix\nmissed:\n\n1. A new `build-and-test-pyscamp-modern` job that explicitly tests\n   pyscamp on Python 3.10 / 3.12 / 3.13 against both NumPy 1.x and\n   NumPy 2.x.  This catches:\n     - the distutils import in setup.py (Python >= 3.12 + setuptools >= 74),\n     - the #129 numpy>=2 selfjoin regression,\n     - any future ABI/API breakage as the conda-forge migrations land.\n   The cp313 + numpy<2 cell is excluded because numpy 1.x has no cp313 wheels\n   on PyPI.\n\n2. The existing `build-cuda-cli` job is bumped from a single CUDA 11.7\n   build to a matrix over CUDA 11.7 and 12.6, with one extra cell that\n   sets -DSCAMP_DISABLE_THRUST_SORT=ON to exercise the new escape hatch\n   path end-to-end.\n\nAlso bumps the deprecated actions/checkout@v2 / setup-python@v2 /\nJimver/cuda-toolkit@v0.2.6 to current major versions (@v4 / @v5 /\n@v0.2.16) on the touched jobs.\n\n* Allow building against CMake 4.x\n\nCMake 4 hard-removed compatibility with cmake_minimum_required < 3.5.\nThe vendored third_party/gflags submodule still declares an old\nminimum, which now produces:\n\n  CMake Error at third_party/gflags/CMakeLists.txt:73 (cmake_minimum_required):\n    Compatibility with CMake < 3.5 has been removed from CMake.\n\nThis is what motivated the conda-forge pyscamp-gpu-feedstock CUDA 12.9\nPR to add a `cmake <4` cap.  The supported upstream escape hatch is\nto set CMAKE_POLICY_VERSION_MINIMUM, which we do at the top of our\nCMakeLists.txt before any subdirectory is added.  The cap on the\nfeedstock side can now be dropped.\n\nVerified: `cmake -DFORCE_NO_CUDA=1 ..` configures and builds cleanly\non CMake 4.3.2 + GCC, both for the SCAMP CLI (which exercises gflags)\nand for the pyscamp BUILDING_PYSCAMP=ON path.\n\n* Committing clang-format changes\n\n* Replace SCAMP_DISABLE_THRUST_SORT opt-out with cub::DeviceMergeSort\n\nDrops the SCAMP_DISABLE_THRUST_SORT CMake option and host-sort fallback\nintroduced earlier in this branch.  Use cub::DeviceMergeSort::SortKeys\ndirectly for match_gpu_sort() instead of thrust::sort.\n\nWhy: the CCCL preprocessor regression that motivated the opt-out\n(_CCCL_PP_SPLICE_WITH_IMPL1 arg-count mismatch under CUDA 12.9 + GCC 13\nin conda-forge's pinning) lives entirely in the THRUST_NAMESPACE_BEGIN\nexpansion path.  CUB sits at a lower CCCL layer and reaches a different\nnamespace-versioning macro (CUB_NAMESPACE_BEGIN), so this whole class\nof failure is sidestepped without losing GPU-side sort performance --\nthrust::sort dispatches to CUB internally on CUDA anyway.\n\nChanges:\n  - src/core/kernels.cu: drop the <thrust/...> includes and the\n    #ifdef SCAMP_DISABLE_THRUST_SORT machinery in match_gpu_sort().\n    Use the standard CUB two-call temp-storage pattern with\n    cudaMallocAsync / cudaFreeAsync so the whole pipeline stays on the\n    user-supplied stream.  A small SCAMPmatchLess functor wraps the\n    existing HOST_DEVICE_FUNCTION operator< on SCAMPmatch.\n  - CMakeLists.txt: remove the SCAMP_DISABLE_THRUST_SORT option block.\n  - .github/workflows/build-and-test.yml: remove the thrust-sort matrix\n    dimension from build-cuda-cli.  CI now builds the same CUDA 11.7\n    and 12.6 cells but without the now-unused opt-out cell.\n\nVerified locally on the no-CUDA path that this still compiles and that\nthe existing test_pyscamp.py suite passes under both NumPy 1.26.4 and\nNumPy 2.2.6.  The CUB sort path itself was not exercised in the\nsandbox (no nvcc) -- needs a one-time CUDA-box verification before\ntagging v4.0.2.\n\n* ci: fix failing matrix - bump action versions and pre-install build deps\n\nPR #136's first run produced ~30 failures across the matrix.  Three\ndistinct root causes, each fixable in this single workflow file:\n\n1. Stale GitHub Actions versions.  The pre-existing jobs were pinned to\n   actions/checkout@v2 and actions/setup-python@v2 (Node 16 runners),\n   which GitHub-hosted runners now refuse to start.  Visible as the\n   self-hosted build-and-test-cuda job's '[failure] Run actions/checkout@v2'.\n   Bump checkout to @v4 and setup-python to @v5 across all jobs.\n\n2. Wrong pinned third-party action versions:\n     - Jimver/cuda-toolkit@v0.2.16 doesn't know about CUDA 12.6.0\n       (annotation: 'Error: Version not available: 12.6.0').  Bump to\n       @v0.2.30, the last 0.2.x point release with stable 11.7 + 12.6\n       support.\n     - petarpetrovt/setup-sde@v2.3 fetches SDE binaries from a URL that\n       now returns 403 ('Unexpected HTTP response: 403').  Bump to @v4.0,\n       which uses the current Intel SDE distribution endpoint.\n\n3. Missing build deps in the pyscamp jobs (both the pre-existing\n   'build-and-test-cpu-pyscamp' and the new 'build-and-test-pyscamp-modern').\n   setuptools 80 removed the easy_install fallback, so setup.py's\n   'setup_requires=['setuptools_scm']' can no longer ephemerally fetch\n   setuptools_scm at build time.  And '--no-build-isolation' in the new\n   job means pip doesn't auto-provision setuptools/wheel either, which\n   produced the 'error: invalid command bdist_wheel' failure.\n\n   Pre-install setuptools>=70, wheel, setuptools_scm, and cmake in both\n   pyscamp build steps.  Also switch the pre-existing job from the\n   deprecated 'python setup.py sdist; pip install dist/*' invocation to\n   the modern 'pip install . --no-build-isolation' so we don't have two\n   different code paths to maintain.\n\nReproduced 'invalid command bdist_wheel' locally with Python 3.10 and\nUbuntu's stock setuptools 59.6, then verified the fix: with the new\npre-install line the build completes and 'import pyscamp' works.\n\nNote: a pre-existing macos-latest 'Extended Tests SCAMP' runtime test\nfailure on the build-and-test-cpu-cli matrix is also red, but that's a\nseparate test-stability issue (not a build problem) and is independent\nof this PR's substantive changes.  Worth a follow-up issue.\n\n* Get tests working.\n\n* Fix CI failures: Windows chrono, macOS arm64 AVX, pip env, SDE version\n\n- Add <chrono> include to scamp_interface.cpp; MSVC does not pull it in\n  transitively like GCC/Clang. Also fix %lu -> %zu for devices.size()\n  to avoid MSVC format-string warnings on Windows x64.\n- Guard AVX/AVX2 kernel subdirs and link targets behind an x86\n  processor check in cpu_kernel/CMakeLists.txt so the distributable\n  build no longer tries to compile x86 intrinsics on macOS arm64\n  (macos-latest is now Apple Silicon / aarch64).\n- Reorder kernel_dispatcher.cpp includes so cpu_features_macros.h is\n  included before the avx dispatch headers, then gate those headers on\n  CPU_FEATURES_ARCH_X86. Non-x86 falls through to the baseline kernel.\n- Add actions/setup-python to build-and-test-cpu-cli so macOS no\n  longer hits the PEP 668 externally-managed-environment pip error.\n- Bump Intel SDE version 9.27.0 -> 9.33.0 for Ubuntu 24.04 compat.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Bump cpu_features v0.7.0 -> v0.10.1 for macOS arm64 support\n\nv0.10.1 adds impl_aarch64_macos_or_iphone.c so the library builds and\nlinks correctly on Apple Silicon. The old v0.7.0 only had\nimpl_aarch64_linux_or_android.c, which left GetAarch64Info undefined\non macOS arm64 and broke the list_cpu_features tool at link time.\n\nBumping rather than skipping cpu_features on non-x86 preserves the\ncpu_features dispatch infrastructure on arm64 for future NEON support.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Fix cpu_features target alias: CpuFeature:: -> CpuFeatures::\n\ncpu_features v0.10.1 renamed the CMake alias from CpuFeature::cpu_features\n(v0.7.0) to CpuFeatures::cpu_features, breaking the distributable build\non all platforms with a target-not-found CMake error.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Consolidate pyscamp CI matrix; harden CUDA arch support\n\nCI: merge build-and-test-cpu-pyscamp and build-and-test-pyscamp-modern\ninto a single build-and-test-pyscamp job (14 jobs, down from 16).\nThe combined matrix covers ubuntu/macos/windows × py3.10/3.12/3.13 ×\nnumpy1/2 × redistrib OFF/ON with strategic excludes:\n- Windows limited to py3.13 + numpy2 (both redistrib values)\n- Redistrib ON only tested at py3.13 to keep job count down\n- numpy1 excluded for py3.13 (no cp313 wheels on PyPI)\nAdopts the modern job's pinned-numpy build pattern throughout.\n\ncmake: rewrite set_cuda_architectures() with correct version gating:\n- Maxwell (SM 5.x): now gated on VERSION_LESS \"13.0\" (was \"12\"); they\n  are deprecated in CUDA 12 but still compile — only removed in CUDA 13\n- Add Blackwell (SM 10.0) for CUDA >= 12.6\n- Rebuild list from scratch instead of appending to placeholder\n- Use COMPARE NATURAL sort so SM 89/90/100 order correctly\n- Add per-family comments documenting the CUDA version at which each\n  architecture was introduced or deprecated\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Fix Blackwell SM 100 gate: CUDA 12.8 not 12.6\n\ncompute_100 is not supported by CUDA 12.6; nvcc fatal error confirms\nSM 100 (Blackwell) requires CUDA >= 12.8.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Fix Windows redistrib build hang: bypass Eigen CMakeLists, exclude cl+ON\n\nEigen's blas/CMakeLists.txt unconditionally calls check_language(Fortran)\nregardless of BUILD_TESTING. On Windows without a Fortran compiler this\nstalled cmake for 90+ minutes. The root cause is that Eigen always adds\nits blas/ and lapack/ subdirectories, which are cmake-evaluated even with\nEXCLUDE_FROM_ALL.\n\nFix: create a header-only IMPORTED INTERFACE target for Eigen instead of\nusing add_subdirectory(). SCAMP only uses Eigen for SIMD intrinsics (pure\nheaders), so no compiled Eigen targets are needed and the full\nCMakeLists.txt evaluation can be skipped entirely. Confirmed no Fortran/\nBLAS/LAPACK search occurs after this change.\n\nAlso exclude windows+cl+redistrib=ON from build-and-test-cpu-cli. This\ncombination was already failing before this PR (1h36m, run 25286043535).\nWindows redistrib builds are covered by the pyscamp Windows job.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Re-enable windows+cl+redistrib=ON in build-and-test-cpu-cli\n\nThe Eigen Fortran hang that caused this to take 90+ minutes is now\nfixed by bypassing Eigen's CMakeLists.txt entirely. Re-add the matrix\ncell now that the root cause is resolved.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Restore BUILD_TESTING CACHE FORCE to suppress cpu_features googletest\n\nWhen BUILD_CLIENT_SERVER=1 (Docker build), gRPC is processed before\nsrc/core and sets BUILD_TESTING=ON in the CMake CACHE. cpu_features\nthen inherits BUILD_TESTING=ON and tries to download googletest, which\nfails with \"CMake step for googletest failed: no such file or directory\".\n\nThe previous set(BUILD_TESTING OFF) (normal variable) was removed when\nswitching to the Eigen IMPORTED INTERFACE approach, inadvertently\nbreaking the Docker build. Restore it as CACHE FORCE so it overrides\nwhatever gRPC left in the cache before cpu_features is configured.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Modernize Docker: CUDA 12.9 + Ubuntu 24.04, bump action versions\n\n- Bump base/runtime images from cuda:12.3.1-ubuntu20.04 to\n  cuda:12.9.0-ubuntu24.04\n- Drop python3-pip + pip install cmake: Ubuntu 24.04 (Noble) ships\n  cmake 3.28 via apt, well above our 3.18 minimum\n- Drop golang-go: not needed for C++ gRPC builds via cmake\n- Use named stage 'base' instead of index '0' in COPY --from\n- Add rm -rf /var/lib/apt/lists/* to keep image layers lean\n- Bump workflow actions: checkout@v2->v4, login-action@v1->v3,\n  metadata-action@v3->v5, build-push-action@v2->v6\n- Add explicit context: . to build-push-action\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Modernize publish and release workflows\n\npython-publish.yml + python-publish-test.yml:\n- checkout@v2 -> @v4, setup-python@v2 -> @v5\n- pypa/gh-action-pypi-publish: pinned SHA -> release/v1 (recommended tag)\n- Drop deprecated 'user: __token__' field\n- Rename repository_url -> repository-url, skip_existing -> skip-existing\n- Add fetch-depth: 0 to python-publish.yml (was missing; setuptools_scm\n  needs full tag history to produce the correct release version)\n\ndocker-build-and-push.yml:\n- Add missing checkout step with submodules: recursive (the release\n  workflow had no checkout at all, so the build context was empty)\n- login-action@v1 -> @v3, metadata-action@v3 -> @v5,\n  build-push-action@v2 -> @v6\n- Add explicit context: .\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Fix Dockerfile: combine apt-get update and install in one RUN\n\nSplitting update and install across separate RUN layers causes the\npackage lists to be wiped (rm -rf /var/lib/apt/lists/*) before the\ninstall step, resulting in \"Unable to locate package\" errors.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: github-actions <41898282+github-actions[bot]@users.noreply.github.com>\nCo-authored-by: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-03T21:23:54-07:00",
          "tree_id": "007f4ec462ad5e0225403f6a1211078fd081206e",
          "url": "https://github.com/zpzim/SCAMP/commit/aaab3a2d072706333fa3f89193af337108d22d4f"
        },
        "date": 1777869037487,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.1292499799999973,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.6297712100000012,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0015625 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8215053099999977,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.479759610000002,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003125 s\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Zach Zimmerman",
            "username": "zpzim",
            "email": "zpzimmerman@gmail.com"
          },
          "committer": {
            "name": "Zach Zimmerman",
            "username": "zpzim",
            "email": "zpzimmerman@gmail.com"
          },
          "id": "e6c5ed48b8d3e41b6f9715230b16d56c028e75d7",
          "message": "Fix Windows benchmark builds: HAVE_STD_REGEX pre-cache + disable Werror\n\nTwo separate Windows failures after the v1.9.5 bump:\n\n1. cl (MSVC): benchmark's regex backend detection compiles a snippet\n   that gates on __cplusplus, but MSVC reports __cplusplus as 199711L\n   unless /Zc:__cplusplus is set. All three regex backends (std,\n   gnu_posix, posix) report 'failed' and benchmark errors out with\n   \"Failed to determine the source files for the regular expression\n   backend\". Pre-set HAVE_STD_REGEX in the cache for any MSVC toolchain\n   to skip the detection (std::regex is always available).\n\n2. clang-cl: SCAMP's global CMAKE_CXX_FLAGS_RELEASE adds Linux-style\n   flags like -O3 which clang-cl warns about as unused arguments\n   ('-Wunused-command-line-argument'). Combined with benchmark's\n   default BENCHMARK_ENABLE_WERROR=ON, these warnings become hard\n   errors. Disable BENCHMARK_ENABLE_WERROR so benchmark builds tolerate\n   incidental flag mismatches with our parent project's flags.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-08T15:03:29Z",
          "url": "https://github.com/zpzim/SCAMP/commit/e6c5ed48b8d3e41b6f9715230b16d56c028e75d7"
        },
        "date": 1778252981161,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.1233041299999968,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.777094210000007,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.00625 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.9643990199999962,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.8374438000000737,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}