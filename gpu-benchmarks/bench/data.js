window.BENCHMARK_DATA = {
  "lastUpdate": 1659624999461,
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
      }
    ]
  }
}