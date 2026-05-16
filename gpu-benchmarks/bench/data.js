window.BENCHMARK_DATA = {
  "lastUpdate": 1778959632441,
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
        "date": 1655575032711,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7745013738982379,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0292439085 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7244187204050831,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0289390879 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.0105864127981476,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029070518499999996 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.442040745099075,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029146927200000006 s\nthreads: 1"
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
        "date": 1655586498101,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7752072755945847,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029366723299999998 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7245093896053731,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029239750999999998 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.0110640415921806,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.02931176579999999 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4419876674073748,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029274034799999994 s\nthreads: 1"
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
        "date": 1659624987754,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7670207266928628,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.021756981 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7163760951021686,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.02113191020000001 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.002786505012773,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.021183625100000002 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4337444199016318,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.021046547700000008 s\nthreads: 1"
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
        "date": 1704734108156,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7740767720999429,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.028379305000000004 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7240224124005181,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.028319570099999997 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.010907543900248,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.028355510699999996 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4413608590999503,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.028438400399999987 s\nthreads: 1"
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
        "date": 1704775509100,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7738590107997879,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0284369109 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.724107754000579,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.028512309900000005 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.010767740600568,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0284816273 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.441256782200071,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0283662404 s\nthreads: 1"
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
          "id": "91159aa6695f74f9af4533f329068a6f24bc206c",
          "message": "Switch PyPI publishing to OIDC trusted publishing\n\nAPI tokens created before 2FA was enabled are invalidated by PyPI.\nRather than rotating the token, switch to trusted publishing (OIDC):\nPyPI trusts GitHub Actions directly, no API token or secret needed.\n\nChanges:\n- Add permissions.id-token: write to both publish jobs\n- Remove password: ${{ secrets.TEST_PYPI_API_TOKEN / PYPI_API_TOKEN }}\n\nTo activate, add a trusted publisher on Test PyPI and PyPI:\n  Publisher: GitHub Actions\n  Owner: zpzim\n  Repository: SCAMP\n  Workflow: python-publish-test.yml / python-publish.yml\n  Environment: (leave blank)\n\nSee: https://docs.pypi.org/trusted-publishers/\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-04T05:14:47Z",
          "url": "https://github.com/zpzim/SCAMP/commit/91159aa6695f74f9af4533f329068a6f24bc206c"
        },
        "date": 1777872273769,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7706807216000016,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0310737285 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7301748823996604,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.031067490799999996 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.0288286207000055,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0309208853 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4399823265997838,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030801450500000004 s\nthreads: 1"
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
          "id": "e66e09839b84ff07685565a705b66a93773a81d9",
          "message": "Bump Google Benchmark v1.9.0 -> v1.9.5\n\nv1.9.5 contains the upstream fix for the Windows/ClangCL build failure:\n\"Fix CXX feature check when try_run compilation fails\" (PR #2046).\nThe regex backend detection used check_cxx_source_runs() which would\ncompile but not run under Visual Studio generators, causing the benchmark\ncmake configure to error out with \"Failed to determine the source files\nfor the regular expression backend\".\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-04T15:08:54Z",
          "url": "https://github.com/zpzim/SCAMP/commit/e66e09839b84ff07685565a705b66a93773a81d9"
        },
        "date": 1777910042374,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7703090978000546,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030912506600000007 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7297736189000716,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0310418829 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.0290508318998035,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.031361776900000006 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4384183278001728,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029467007200000016 s\nthreads: 1"
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
        "date": 1778255176829,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7706460316025187,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.031126937600000005 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7296907738025766,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030825587500000008 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.02860345429508,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.03087852149999999 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4384429736994206,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0292762454 s\nthreads: 1"
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
          "id": "ef4d0cdd48621de7ec0d70abd9286cf0facc95c3",
          "message": "Fix benchmark/docs/PyPI CI issues (#137)\n\n* Fix benchmark build with CMake 4: disable GoogleTest download, bump to v1.9.0\n\nBENCHMARK_DOWNLOAD_DEPENDENCIES caused Google Benchmark to fetch and\nconfigure GoogleTest in a cmake subprocess. That subprocess does not\ninherit CMAKE_POLICY_VERSION_MINIMUM, so CMake 4.x rejected GoogleTest's\nold cmake_minimum_required with \"Compatibility with CMake < 3.5 has\nbeen removed\".\n\nFix: set BENCHMARK_ENABLE_TESTING OFF so GoogleTest is never needed,\nand bump Google Benchmark v1.6.1 -> v1.9.0 which has native CMake 4\ncompatibility.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Fix RTD docs build: FORCE_NO_CUDA, setuptools-scm, mock pyscamp import\n\n- Add FORCE_NO_CUDA=1 to the RTD build environment so cmake does not\n  attempt CUDA detection in the Read the Docs build sandbox\n- Add setuptools-scm to docs/requirements.txt so it is available before\n  pip install . invokes setup.py (which imports it for versioning)\n- Add autodoc_mock_imports = ['pyscamp'] to conf.py so Sphinx can build\n  even if the C extension fails to compile or import; pyscamp is a\n  pybind11 module that requires a full cmake build to import\n- Bump release string from 4.0.0 to 4.0.1\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Remove FORCE_NO_CUDA from .readthedocs.yaml (not supported)\n\nThe build.environment key does not work in this context. cmake will\ndetect the absence of CUDA naturally and fall back to a CPU-only build.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Drop pyscamp pip install from RTD build\n\ncmake installed via pip does not land on PATH for subprocess calls made\nby setup.py, causing 'cmake --version' to return exit code 1 and the\nwheel build to fail. Since pyscamp is a C extension with no pure-Python\nfallback, building it in the RTD sandbox is fragile.\n\nautodoc_mock_imports = ['pyscamp'] (added in the previous commit) lets\nSphinx generate the docs without importing the compiled module, so the\npip install step is unnecessary. Also drop requirements.txt (cmake) from\nthe install list since nothing else in the docs build needs it.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Fix RTD API docs: replace autodoc mock with pure-Python stub\n\nautodoc_mock_imports produces empty documentation because mock objects\nhave no docstrings. Instead, add docs/pyscamp.py — a pure-Python stub\nthat contains the real function signatures and docstrings transcribed\nfrom src/python/SCAMP_python.cpp. Sphinx imports the stub (no cmake,\nno C++ compiler needed) and generates complete API documentation.\n\nconf.py now inserts docs/ at the front of sys.path so the stub takes\nprecedence over any installed compiled pyscamp binary.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Switch PyPI publishing to OIDC trusted publishing\n\nAPI tokens created before 2FA was enabled are invalidated by PyPI.\nRather than rotating the token, switch to trusted publishing (OIDC):\nPyPI trusts GitHub Actions directly, no API token or secret needed.\n\nChanges:\n- Add permissions.id-token: write to both publish jobs\n- Remove password: ${{ secrets.TEST_PYPI_API_TOKEN / PYPI_API_TOKEN }}\n\nTo activate, add a trusted publisher on Test PyPI and PyPI:\n  Publisher: GitHub Actions\n  Owner: zpzim\n  Repository: SCAMP\n  Workflow: python-publish-test.yml / python-publish.yml\n  Environment: (leave blank)\n\nSee: https://docs.pypi.org/trusted-publishers/\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Bump Google Benchmark v1.9.0 -> v1.9.5\n\nv1.9.5 contains the upstream fix for the Windows/ClangCL build failure:\n\"Fix CXX feature check when try_run compilation fails\" (PR #2046).\nThe regex backend detection used check_cxx_source_runs() which would\ncompile but not run under Visual Studio generators, causing the benchmark\ncmake configure to error out with \"Failed to determine the source files\nfor the regular expression backend\".\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Make Docker build resilient to Ubuntu mirror flakiness\n\nThe previous Dockerfile ran 'apt-get upgrade -y' which pulled in dozens\nof unrelated packages (llvm-18 toolchain, libicu, libpython3.12, libxml2,\netc.). On a recent CI run, archive.ubuntu.com was intermittently\nunreachable and the build failed after 8 minutes of slow downloads\nbecause one of these incidental packages couldn't be fetched.\n\nChanges:\n- Drop 'apt-get upgrade -y'. The cuda:12.9.0-devel-ubuntu24.04 base image\n  is already current enough; we don't need to upgrade everything just to\n  install three packages.\n- Add 'Acquire::Retries \"3\"' so individual package fetches retry on\n  transient connection failures instead of failing the whole build.\n- Add '--no-install-recommends' to keep the install minimal.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>\n\n* Fix Windows benchmark builds: HAVE_STD_REGEX pre-cache + disable Werror\n\nTwo separate Windows failures after the v1.9.5 bump:\n\n1. cl (MSVC): benchmark's regex backend detection compiles a snippet\n   that gates on __cplusplus, but MSVC reports __cplusplus as 199711L\n   unless /Zc:__cplusplus is set. All three regex backends (std,\n   gnu_posix, posix) report 'failed' and benchmark errors out with\n   \"Failed to determine the source files for the regular expression\n   backend\". Pre-set HAVE_STD_REGEX in the cache for any MSVC toolchain\n   to skip the detection (std::regex is always available).\n\n2. clang-cl: SCAMP's global CMAKE_CXX_FLAGS_RELEASE adds Linux-style\n   flags like -O3 which clang-cl warns about as unused arguments\n   ('-Wunused-command-line-argument'). Combined with benchmark's\n   default BENCHMARK_ENABLE_WERROR=ON, these warnings become hard\n   errors. Disable BENCHMARK_ENABLE_WERROR so benchmark builds tolerate\n   incidental flag mismatches with our parent project's flags.\n\nCo-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-08T09:01:00-07:00",
          "tree_id": "ec6fa505b020d0bec16c105ebeffbff133a4a830",
          "url": "https://github.com/zpzim/SCAMP/commit/ef4d0cdd48621de7ec0d70abd9286cf0facc95c3"
        },
        "date": 1778256474707,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7698631566017866,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0300836393 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7292141504993197,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030206176600000002 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.0281877461005933,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0303862737 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4384439358022063,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029264052799999996 s\nthreads: 1"
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
          "id": "8d56f8ef127b7ff6be510180bef53a80fab9f663",
          "message": "Fix CUDA 13.0 compatibility + harden arch support matrix (#138)\n\n* Fix CUDA 13.0 compatibility + update arch support matrix\n\nArchitectures removed in CUDA 13.0: Pascal (SM 6.x) and Volta (SM 7.0/7.2)\nalongside Maxwell — gate all with VERSION_LESS \"13.0\". Fixes nvcc fatal error\nwhen building with CUDA 13.\n\ncuFFT constants removed in CUDA 13.0: CUFFT_PARSE_ERROR, LICENSE_ERROR,\nINCOMPLETE_PARAMETER_LIST — guard with #ifdef CUFFT_PARSE_ERROR.\n\nUpdate CMAKE_CUDA_ARCHITECTURES placeholder from SM 60/61 to SM 75, which\nis valid from CUDA 10.0 through CUDA 13+, avoiding enable_language(CUDA)\nfailures with CUDA 13.\n\nCorrections and additions to set_cuda_architectures() (per CUDA release notes):\n- SM 87 (Jetson Orin): corrected introduction to CUDA 11.4 (was 11.5)\n- SM 100/101 (Blackwell DC) + SM 120 (Blackwell consumer): added at CUDA 12.8\n- SM 103, SM 121: added at CUDA 12.9\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* Add libcub-dev to Docker build dependencies\n\nBlackwell arch targets (SM 100+) trigger CUB code paths that require\nconsistent libcub/libcccl header versions. Without libcub-dev installed\nas an explicit package, the devel image's bundled CUB headers and CCCL\npreprocessor headers can be from different sub-releases, causing:\n  error: macro \"_CCCL_PP_SPLICE_WITH_IMPL1\" passed 3 arguments, but takes just 2\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* Bump build-cuda-cli to CUDA 12.9.1, add libcub-dev on Linux\n\n12.9.1 is the latest point release and exercises Blackwell arch targets\n(SM 100+). libcub-dev ensures consistent CUB/CCCL header versions on the\nGitHub-hosted Ubuntu runner, matching the fix already applied to the\nDockerfile and local self-hosted runner.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* Work around CCCL 2.8.x arch token overflow (NVIDIA/cccl#4967)\n\nCUDA 12.9.x ships CCCL 2.8.x which overflows when too many GPU arch\nnumbers are concatenated into the versioned inline namespace name. Fixed\nin CCCL 3.0.0 (CUDA 13.0+), not backported to 2.8.x.\n\nOn CUDA 12.8-12.x, limit the arch list to 11 entries:\n  60 61 70 75 80 86 87 89 90 100 120\n- Maxwell (SM 5x) excluded (deprecated; removed in CUDA 13)\n- SM 62 (Tegra X2) and SM 72 (Jetson Xavier) excluded (embedded-only)\n- Blackwell limited to SM 100 + SM 120 (minor variants SM 101/103/121\n  added back on CUDA 13.0+ where the bug is fixed)\n- Pascal (SM 60/61) and Volta (SM 70) retained\n\nOn CUDA 13.0+: full Blackwell variant set included; CCCL 3.0.0 handles\nthe larger list correctly.\n\nAlso revert the libcub-dev workarounds (Dockerfile, CI) added under the\nwrong diagnosis — the arch count reduction is the real fix.\nBump build-cuda-cli CI to CUDA 12.9.1 (latest 12.x point release).\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-09T12:30:50-07:00",
          "tree_id": "b1354880ba8e65867a95388c5d98a14b523531a2",
          "url": "https://github.com/zpzim/SCAMP/commit/8d56f8ef127b7ff6be510180bef53a80fab9f663"
        },
        "date": 1778355227984,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7697543391026557,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0301207816 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7290602641995065,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030171190500000007 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.0282619536970743,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.03044743179999998 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4380946389050222,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.028933337000000003 s\nthreads: 1"
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
          "id": "0c5d3640118336af653900b3cb5a60f9beef8dcd",
          "message": "Fix MSVC C4505 warning + compute_101 unsupported arch on CUDA 13.0 (#141)\n\n* Fix MSVC C4505 warning for _cudaGetErrorEnum in qt_helper.h\n\nstatic functions have internal linkage, so MSVC warns C4505 when a\ntranslation unit includes the header but never calls CHECK_CUFFT_ERRORS.\ninline has external linkage and is the correct specifier for a\nutility function defined in a header.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* Fix compute_101 error on CUDA 13.0+; add multi-version CUDA build CI\n\nsm_101 (Jetson Thor) was renamed to sm_110 in CUDA 13.0, causing\n'nvcc fatal: Unsupported gpu architecture compute_101' when building\nwith the CUDA 13.0 conda-forge package. Replace 101 with 110 in the\n>= 13.0 arch list and add sm_103 / sm_121 (introduced CUDA 12.9,\nnow safe to include since the CCCL arity bug is fixed in 13.0).\n\nAlso add a build-cuda-versions CI job that compiles with CUDA 12.6.0,\n12.8.0, and 13.0.0 on both Linux (g++) and Windows (cl), covering the\nkey architectural breakpoints not already tested by build-cuda-cli\n(which tests 12.9.1 across compilers).\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-09T17:10:31-07:00",
          "tree_id": "608a45382fca75d371903b33e81bf1531832051d",
          "url": "https://github.com/zpzim/SCAMP/commit/0c5d3640118336af653900b3cb5a60f9beef8dcd"
        },
        "date": 1778374013832,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.771531485801097,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030967146100000002 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7296868739009369,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030703178399999996 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.028807978000259,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.03110502110000001 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4388924397004303,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.02978437079999998 s\nthreads: 1"
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
          "id": "22bda6e274db48f3f70dd5e763c99295c70e02d7",
          "message": "Fix exclusion zone for small window sizes and improve docs/comments (#140)\n\n* Fix exclusion zone for small window sizes (issue #135)\n\nWith m<4, floor(m/4)=0 meant no exclusion zone was applied. Each\nsubsequence then matched itself (corr=1.0) as its nearest neighbor,\nproducing Euclidean distance 0.0 everywhere.\n\nFix get_exclusion() to use ceiling division -- (m+3)/4 -- so m=3 gets\nexclusion=1 instead of 0, correctly excluding the trivial self-match.\n\nAlso apply the ca75a21 off-by-one fix for transposed (lower) tiles:\nthe transposed geometry shifts the effective exclusion boundary by one\ndiagonal, so exclusion_upper is reduced by 1 in get_exclusion_for_self_join\nand get_exclusion_for_ab_join to avoid missing the corner value.\n\nFix the same floor-division bug in the Python reference implementation\n(distance_matrix_fast.py) and add correctness tests for m=3 and m=4.\n\n* Document MATRIX_SUMMARY threshold behaviour (issue #134)\n\nWith the default threshold of 0, cells whose pooling window contains only\nnegative correlations are left as NaN. Document this and note that\nthreshold=-1 guarantees all cells are filled.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-11T08:48:58-07:00",
          "tree_id": "a52da91da6ee2908adf7e83756120a852c1c9059",
          "url": "https://github.com/zpzim/SCAMP/commit/22bda6e274db48f3f70dd5e763c99295c70e02d7"
        },
        "date": 1778516677108,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.770131749601569,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030525071199999997 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7291029710089788,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030139425799999998 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.0282945672981443,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030309888900000005 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4379574737977237,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.028743066400000017 s\nthreads: 1"
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
          "id": "9a5278035dd0a02107b1be6923a7c584b4680a2b",
          "message": "Add allow_trivial_match to pyscamp ab-joins; consolidate CUDA CI matrix (#142)\n\n* Add allow_trivial_match kwarg to pyscamp ab-join functions (issue #132)\n\nExposes the underlying SCAMPArgs.is_aligned option (CLI: --aligned) to\npyscamp's ab-join APIs. When False, treats a and b as aligned segments\nof the same series and applies an exclusion zone equivalent to a\nself-join, filtering trivial near-diagonal matches.\n\n- Default True for ab-joins (preserves existing pyscamp behaviour: no\n  exclusion zone, all subsequence pairs considered).\n- Passing the kwarg to a self-join raises ValueError, since self-joins\n  already exclude trivial matches by construction.\n\nAdded tests verifying:\n  - mp.abjoin(a, a, m) returns ~1.0 everywhere by default (trivial self-\n    match diagonal is included).\n  - mp.abjoin(a, a, m, allow_trivial_match=False) exactly matches\n    mp.selfjoin(a, m).\n  - Passing allow_trivial_match to a self-join raises ValueError.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* Committing clang-format changes\n\n* Consolidate CUDA build CI into a single matrix to fit cache budget\n\nPreviously we had two CUDA build jobs that together pinned six\n(OS, CUDA) cache entries (~24GB) against GitHub's 10GB per-repo cache:\n  - build-cuda-cli covered 12.9.1 across g++/clang++/cl (2 cache entries)\n  - build-cuda-versions covered 12.6.0/12.8.0/13.0.0 across g++/cl\n    (6 cache entries, with 12.6.0 already overlapping build-cuda-cli's\n     coverage and racing for the same Jimver/cuda-toolkit artifact name)\n\nMerge both into build-cuda-versions and prune to the breakpoints that\nmatter for arch/CCCL coverage: 12.8.0 (CCCL 2.8.x arch-token bug) and\n13.0.0 (CCCL 3.0 fix plus sm_110 / Blackwell rename). 12.9.1 sits\nbetween these and the version gates in SCAMPMacros.cmake mean its\narch list is a subset of what 13.0.0 exercises -- dropping it loses no\nunique coverage. Keep ubuntu+clang++ at 13.0.0 only as a smoke test for\nclang as the host compiler; nvcc owns CUDA codegen so testing it on\nevery CUDA version is low-value.\n\nResult: 5 build jobs, 4 cache entries (~16GB). Still above the 10GB\nbudget but a 33% reduction, and no overlapping (OS, CUDA) entries\nbetween jobs -- which also eliminates the parallel Jimver/cuda-toolkit\nartifact-name 409 conflicts that were popping up on 12.6.0.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: github-actions <41898282+github-actions[bot]@users.noreply.github.com>",
          "timestamp": "2026-05-16T12:24:15-07:00",
          "tree_id": "7cb6ab5dc776bd9c1673b6036ed1e8f908387216",
          "url": "https://github.com/zpzim/SCAMP/commit/9a5278035dd0a02107b1be6923a7c584b4680a2b"
        },
        "date": 1778959631062,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/524288",
            "value": 0.7708648418076336,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.031008067299999997 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/524288",
            "value": 0.7298798291012645,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.030814472699999994 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/524288",
            "value": 2.029373015300371,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.03135451619999998 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/524288",
            "value": 1.4390538105973973,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.029714479400000004 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}