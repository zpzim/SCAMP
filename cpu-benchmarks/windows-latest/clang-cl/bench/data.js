window.BENCHMARK_DATA = {
  "lastUpdate": 1642572837701,
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
          "id": "4e68ad217f232b4da0d865dd9d7cdc80e40cb598",
          "message": "Remove cache from cpu-benchmark Action",
          "timestamp": "2022-01-18T20:23:55-08:00",
          "tree_id": "76c0fd748ddf6ee8c763d04739003491ae0d35b5",
          "url": "https://github.com/zpzim/SCAMP/commit/4e68ad217f232b4da0d865dd9d7cdc80e40cb598"
        },
        "date": 1642566968457,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1280843570.000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 5136257300.000125,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 20636569699.99988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 978748229.9999965,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3882610699.9999866,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 15208174999.999983,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 511245699.99999493,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2057954390.000009,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 9375000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 7985922600.000094,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 424773020.00002056,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1493023399.9999928,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 12500000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5560336500.000175,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 2040722240.0000138,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 8123398100.000031,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 32663745999.999947,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1545053020.0000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 6166366400.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 24676595399.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
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
          "id": "84d53a927f3ff05ee9d70e2fc9c67993a083213e",
          "message": "Add benchmarks for CPU Kernels (#79)\n\n* Add benchmarks for CPU kernel performance.\r\n* Add Action to monitor performance.",
          "timestamp": "2022-01-18T21:58:45-08:00",
          "tree_id": "25519506b8be41416e640dc47737a8353d281947",
          "url": "https://github.com/zpzim/SCAMP/commit/84d53a927f3ff05ee9d70e2fc9c67993a083213e"
        },
        "date": 1642572832666,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 5306345600.000214,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 21262843799.999928,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 84618091000.00005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 3919059000.000061,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 15562529899.999846,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 62507961900.00005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 4659645599.999976,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 18486210000.000027,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 73772346599.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 3432638400.0000873,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 13609032200.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 54520947800.00016,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 5919992199.999797,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 23778485600.00007,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 94981786099.99991,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 4330421699.999988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 17563424499.99988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 70012828599.99992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}