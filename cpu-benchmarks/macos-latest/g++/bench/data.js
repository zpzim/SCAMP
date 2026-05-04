window.BENCHMARK_DATA = {
  "lastUpdate": 1777908384179,
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
        "date": 1655575097823,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.245291303800002,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019091 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.7305221855000014,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018606999999999999 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.319239746400001,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018907 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.6912426320000122,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.002023000000000011 s\nthreads: 1"
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
        "date": 1655586343083,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.9397992732000603,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0018926 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.574171920200024,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0021248 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.1000136890999783,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.001897 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.574480117999883,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0019599999999999895 s\nthreads: 1"
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
        "date": 1659625022416,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.029546320999998,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0020047999999999997 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.6498834717000023,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0020051999999999995 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.2214413198999978,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019199000000000009 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.7582445489999827,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.002035000000000009 s\nthreads: 1"
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
        "date": 1690849230826,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.965132767099999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019268999999999998 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.5972585501000026,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019643 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.1020548316000032,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0019990000000000008 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.7274813579999773,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.002041000000000001 s\nthreads: 1"
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
        "date": 1704703953154,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.8131382473000031,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0012431 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.5443082597000057,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0029156 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.0690046287000086,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0012383 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.9123546820000001,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0011731000000000005 s\nthreads: 1"
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
        "date": 1704774464083,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.8422539255999937,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0011798000000000002 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.5058347779000087,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0011872999999999999 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.4810175830999923,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.006926399999999999 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.6900348250001116,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0011859999999999926 s\nthreads: 1"
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
        "date": 1777871951451,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.251096604099996,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0008081000000000005 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.38208663329999126,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0007581999999999977 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8689685250000025,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0007308999999999898 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.9711995830000433,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0007559999999999789 s\nthreads: 1"
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
        "date": 1777908381914,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.1723335625000004,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0008175999999999961 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.3700805208999952,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0008676999999999935 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.9542842666000013,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0007720999999999978 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.288892625000017,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0008590000000000542 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}