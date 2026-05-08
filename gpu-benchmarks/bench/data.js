window.BENCHMARK_DATA = {
  "lastUpdate": 1778255177808,
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
      }
    ]
  }
}