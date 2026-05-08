window.BENCHMARK_DATA = {
  "lastUpdate": 1778256687950,
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
        "date": 1655575549297,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.6668993746046908,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017307366000000005 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.2908660660032183,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016682655999999997 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.4389225324965083,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017358304000000004 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.7697053762967698,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017345934999999993 s\nthreads: 1"
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
        "date": 1655587001146,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.6725292164017447,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017119373000000003 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.29109297939576206,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016856915000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.3970766910002568,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017540803999999994 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.722790054208599,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017357102000000013 s\nthreads: 1"
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
        "date": 1659625510368,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.6717497295932844,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016881494000000005 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.29083884770516305,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016388580000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.3954397616907954,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017085878999999998 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.7182085000909866,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017090126999999996 s\nthreads: 1"
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
        "date": 1704733801221,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.6697879581995949,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016829839 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.29196930870020876,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016270017000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.4001923618001455,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016641635000000001 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.7239930303992879,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016796957000000002 s\nthreads: 1"
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
        "date": 1704776051820,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.6700734545011073,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016607384999999999 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.29167923820059516,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016266531 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.4009169298005872,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016813923999999994 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.7233991517001415,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016748887000000005 s\nthreads: 1"
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
        "date": 1777872067190,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.48383499099982147,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018851920000000002 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.22981214971001462,
            "unit": "s/iter",
            "extra": "iterations: 100\ncpu: 0.0018354412700000004 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8649228394002421,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.001893313699999999 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.4584031226997467,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019315530000000025 s\nthreads: 1"
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
        "date": 1777909579923,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.483161209699756,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018296797000000003 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.22947365327003355,
            "unit": "s/iter",
            "extra": "iterations: 100\ncpu: 0.0018232367399999999 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8636116523994133,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018817405000000008 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.4536333526004455,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018957139000000012 s\nthreads: 1"
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
        "date": 1778255436083,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.48541717139887625,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019070626 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.23010656768048648,
            "unit": "s/iter",
            "extra": "iterations: 100\ncpu: 0.00184101909 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8634834854979999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018994307000000017 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.455446191795636,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019088744999999963 s\nthreads: 1"
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
        "date": 1778256686528,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.4840091287973337,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.001906745 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.23038496520020998,
            "unit": "s/iter",
            "extra": "iterations: 100\ncpu: 0.0018312896700000004 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8668330246000551,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.001889822100000002 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.4545652907982003,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018755919999999982 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}