window.BENCHMARK_DATA = {
  "lastUpdate": 1655575044191,
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
      }
    ]
  }
}