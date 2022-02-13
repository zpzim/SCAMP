window.BENCHMARK_DATA = {
  "lastUpdate": 1644780150793,
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
          "id": "a09aed826cc4867f4754138678270bb8475afedd",
          "message": "Refactor benchmarks suite, add gpu benchmarks. (#88)\n\n* Refactor Benchmark library and add GPU benchmarks\r\n\r\n* Reduce the number of CPU benchmarks computed.",
          "timestamp": "2022-02-13T11:17:05-08:00",
          "tree_id": "7059afba734693fece8c8570a08d8b1c0aced0f4",
          "url": "https://github.com/zpzim/SCAMP/commit/a09aed826cc4867f4754138678270bb8475afedd"
        },
        "date": 1644780139179,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3436079255.770892,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 71211620 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2938054548.110813,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62837513.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7926748527.213931,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 65405683.00000002 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5770451938.267797,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62616296.999999985 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}