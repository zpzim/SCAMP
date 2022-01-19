window.BENCHMARK_DATA = {
  "lastUpdate": 1642565410215,
  "repoUrl": "https://github.com/zpzim/SCAMP",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "zpzimmerman@gmail.com",
            "name": "zpzim",
            "username": "zpzim"
          },
          "committer": {
            "email": "zpzimmerman@gmail.com",
            "name": "zpzim",
            "username": "zpzim"
          },
          "distinct": true,
          "id": "72274eb699cbb066fb3b638c09404d47213ab159",
          "message": "Fix benchmark output path on Windows. Also compile benchmarks with clang-cl on Windows.",
          "timestamp": "2022-01-18T20:00:24-08:00",
          "tree_id": "3ded6f12eef7525e3758420a8cfbf5d2a1bbbaaf",
          "url": "https://github.com/zpzim/SCAMP/commit/72274eb699cbb066fb3b638c09404d47213ab159"
        },
        "date": 1642565405693,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1072417789.9999974,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4365694999.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 16838059600.000065,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 819242629.9999965,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3185702499.9999337,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 12507913499.999973,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 424058060.0000044,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1685562279.9999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 10937500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6333414800.000014,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 369291499.9999971,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 9375000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1256690389.9999943,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7812500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4998583700.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1707519500.0000007,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6801634499.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 27069809999.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1274261609.999985,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 5048579299.999801,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 20558469299.999844,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}