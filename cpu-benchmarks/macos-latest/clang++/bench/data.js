window.BENCHMARK_DATA = {
  "lastUpdate": 1778514954167,
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
        "date": 1655575127105,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.9723025481999912,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019368 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.5939502624999932,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019657999999999993 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.1415251150000016,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0020235999999999995 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.704728262000117,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0021729999999999944 s\nthreads: 1"
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
        "date": 1655586013864,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.0559076253000057,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0020214 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.6555322100000012,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0020151 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.235015675700015,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.001989 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.8349569399997563,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0019970000000000127 s\nthreads: 1"
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
        "date": 1659624981769,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.0430934272999992,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019589000000000004 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.6078471536999984,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018739 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.1726081388000011,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.001999899999999999 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.7668920589999857,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0021869999999999945 s\nthreads: 1"
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
        "date": 1690849199293,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.1433742223999956,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019428999999999998 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.8516857705000007,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0022183000000000003 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.484229109499995,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019686999999999994 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.837643049999997,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0020789999999999975 s\nthreads: 1"
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
        "date": 1704703630221,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.3423716732000002,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0024067999999999997 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.8146042611999974,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0031238000000000004 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.694479712500015,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.002608799999999999 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.221782605000044,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0016690000000000038 s\nthreads: 1"
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
        "date": 1704774206163,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.7101588380000067,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0011355 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.42550382479998916,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0011040999999999996 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.0205269863000013,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0011093 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.950189840200005,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0010918000000000004 s\nthreads: 1"
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
        "date": 1777871978929,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.552806670899986,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0012558999999999986 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.5123772791999954,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0009195999999999982 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.0274985000000014,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0008808999999999845 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.385601082999983,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0009660000000000224 s\nthreads: 1"
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
        "date": 1777908323230,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.7431272250000007,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0027761000000000036 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.5485124667000036,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0014998999999999985 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.2970055125000044,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016291000000000055 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.8562352089999195,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0013390000000002011 s\nthreads: 1"
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
        "date": 1778253271174,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.462725158400002,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.001117099999999993 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.4393523542000025,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0009859000000000061 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.040652691599996,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.000886100000000023 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.421362707999947,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0008240000000001579 s\nthreads: 1"
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
        "date": 1778256269599,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.4292136916999993,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0011105000000000032 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.4383388540999931,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0009011999999999909 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.9541778958000009,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0007645999999999931 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.1785768329999655,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0008039999999999159 s\nthreads: 1"
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
        "date": 1778355364662,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.2051990708000062,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0008243999999999918 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.4041659208000056,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.000758100000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.043219799999997,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0009413999999999922 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.5321653340000694,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0008730000000000127 s\nthreads: 1"
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
        "date": 1778372494625,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.1198407583000005,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0008294000000000024 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.3843208790999995,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0007493999999999889 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.9401921500000014,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0008173999999999904 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.1365824169999996,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0009190000000001142 s\nthreads: 1"
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
        "date": 1778514951858,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.3485105500000003,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0012257999999999991 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.4425000041000004,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.000989899999999988 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.156701016599999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0015132000000000144 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.6078985839999973,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0028520000000000767 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}