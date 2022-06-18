window.BENCHMARK_DATA = {
  "lastUpdate": 1655513011641,
  "repoUrl": "https://github.com/zpzim/SCAMP",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "github-actions",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub Actions",
            "username": "actions-user",
            "email": "actions@github.com"
          },
          "id": "ca9ca3c5156be404e3fd33ad842196d31f6c8b89",
          "message": "Committing clang-format changes",
          "timestamp": "2022-06-18T00:32:44Z",
          "url": "https://github.com/zpzim/SCAMP/commit/ca9ca3c5156be404e3fd33ad842196d31f6c8b89"
        },
        "date": 1655513000120,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 0.5945263533969409,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016820535000000004 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 0.2848660804098472,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016114876999999994 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 0.8049787314957939,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016741215999999989 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 1.709340523602441,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0016962230999999993 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}