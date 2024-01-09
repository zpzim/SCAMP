window.BENCHMARK_DATA = {
  "lastUpdate": 1704774065575,
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
        "date": 1655574934561,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.055455386699998,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0028345800000000006 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.6823733179999977,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0028474300000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.469230192399999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.002856470000000001 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.9170487229999935,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0028223000000000137 s\nthreads: 1"
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
          "id": "a16f33c498807b56e0105d5fc23344aaad676015",
          "message": "Improve benchmarking to add stable CPU benchmarks and reduce variance on Github runners. (#118)\n\n* Add a stable version of gcc/clang benchmarks on self-hosted linux box.\r\n\r\n* Update benchmarking to run smaller benchmarks. Output timing information in seconds.\r\n\r\n* Update docs to point to stable benchmark suites.",
          "timestamp": "2022-06-18T10:52:55-07:00",
          "tree_id": "aa3c204e0fb719b7016d86a95d7232bca453e9a2",
          "url": "https://github.com/zpzim/SCAMP/commit/a16f33c498807b56e0105d5fc23344aaad676015"
        },
        "date": 1655574934561,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1.055455386699998,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0028345800000000006 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.6823733179999977,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0028474300000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.469230192399999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.002856470000000001 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.9170487229999935,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0028223000000000137 s\nthreads: 1"
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
        "date": 1655585877291,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.9240956790999973,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003814530000000001 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.5455501721000019,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0038577200000000007 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.205233132899997,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0038544800000000004 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 3.3911252660000173,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0038084000000000173 s\nthreads: 1"
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
        "date": 1659624878945,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.8446446205000029,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0032057799999999997 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.42656851970000387,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.00321013 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.2822360935999995,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003224740000000001 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.6108527279999976,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0035871000000000097 s\nthreads: 1"
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
        "date": 1690849023734,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.952178859899999,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0032310000000000004 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.31676931519999985,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0032489300000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.2072277932999982,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.003177970000000001 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.5841425940000136,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.0030856999999999968 s\nthreads: 1"
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
        "date": 1704703413965,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.7142464469000004,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0024834600000000003 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.2670786133000007,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0024790599999999995 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.7389327300000019,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0025019700000000005 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.1009427267999996,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0024827100000000017 s\nthreads: 1"
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
        "date": 1704774065018,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.7181997421999995,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.00247696 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.2677603723999994,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0025544500000000006 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.7435979602999992,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0025507699999999995 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 2.110027964999998,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.00246098 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}