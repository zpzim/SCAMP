window.BENCHMARK_DATA = {
  "lastUpdate": 1642566343520,
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
      },
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
          "id": "ba94b7e83fc9edd3587b6efc55a192c06a3c74f2",
          "message": "Keep cpu benchmarks in their own directory.",
          "timestamp": "2022-01-18T20:14:31-08:00",
          "tree_id": "4d9486e5c22d935d2aa859db29696539d5a10d75",
          "url": "https://github.com/zpzim/SCAMP/commit/ba94b7e83fc9edd3587b6efc55a192c06a3c74f2"
        },
        "date": 1642566340496,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1018862949.9999933,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4068658699.999901,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15832461899.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 792054669.9999931,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2871533300.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 11845553600.000017,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 399726410.0000052,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1553152339.999997,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5967299799.999978,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 308604549.99999547,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1132248939.9999995,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4212964100.000022,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1599061000.0000062,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1562500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6324452300.000075,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25133690099.999966,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1191011770,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 9375000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4837021399.999912,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 19207125000.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}