window.BENCHMARK_DATA = {
  "lastUpdate": 1655586756042,
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
        "date": 1655575292802,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.5892703285091556,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016911578999999999 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.28541248630499466,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016260911000000003 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8046125133987516,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016813188999999992 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.7257998532964849,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017039196999999993 s\nthreads: 1"
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
        "date": 1655586745040,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.5861764147994108,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016654911 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.285486515390221,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016080752999999997 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8028392515028827,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016781525000000006 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.7228199505014345,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0017042716999999998 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}