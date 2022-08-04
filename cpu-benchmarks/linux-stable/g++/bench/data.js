window.BENCHMARK_DATA = {
  "lastUpdate": 1659625521401,
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
      }
    ]
  }
}