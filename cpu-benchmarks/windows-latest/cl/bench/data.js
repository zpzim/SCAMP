window.BENCHMARK_DATA = {
  "lastUpdate": 1644197666358,
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
        "date": 1642565380755,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 2120908460.0000095,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 8461425799.999915,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 33850828800.000046,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 1586916450.0000124,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1562500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 6323468000.000048,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 25329713699.999958,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1437230840.0000064,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 5696135899.999945,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 22745096900.000134,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 1085165080.0000014,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 4279154199.999994,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 17007118199.999922,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1541376609.9999976,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1562500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6178298999.999925,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24563337099.999897,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1173989579.9999886,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4650781699.999925,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18404466199.99989,
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
        "date": 1642566388867,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 2119482940.0000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 8455047300.000047,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 33742412500,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 1586270379.9999964,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 6394473500.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 25195458499.999973,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1432281370.0000098,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 5698687299.999961,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 22682214799.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 1081455710.0000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 4273966999.9999704,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 17017689000.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1542697579.9999924,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7812500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6132772100.000011,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24407729600.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1162611440.0000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4601474800.000006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18454468000.00002,
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
        "date": 1642566947442,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 2125145539.9999938,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 8470267199.999853,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 33882340299.999897,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 1593617959.9999833,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 6339359699.999931,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 25389625399.999886,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1443513129.9999967,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 5710165100.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 22886468100,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 1087378109.9999862,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7812500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 4283612499.9998903,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 17073935499.999834,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1545974469.9999874,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6164451999.999983,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24491602800.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1171647589.9999976,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4614843800.000017,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18389451500.00018,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
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
        "date": 1642572801608,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 5694946000.0000725,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 22687418800.00016,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 91165969200.00006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 4246919500.0001035,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 16947820299.999876,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 68337886000.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 5215680300.0000305,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 20904782900.0001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 83031572299.99988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 3809727000.0000663,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 15392255700.000078,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 61998956400,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 5868151900.000157,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 23512477499.99993,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 94506470000.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 4391035999.999986,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 17796968500.00005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 70335387800.00003,
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
          "id": "0f8c54a43e8af41588172cf8696e793a43175f5b",
          "message": "Dramatically improve performance of CPU kernels on less sophisticated compilers (#78)\n\n* Force inline CPU kernel methods\r\n\r\n* Add warnings when compiling cpu kernels and inlining fails\r\n\r\n* Optimize CPU kernels to improve the probability that loop vectorization occurs\r\n\r\n* Only check for nan values in the CPU kernels if necessary.\r\n\r\n* Clang will now output whether or not loops were vertorized during compilation.",
          "timestamp": "2022-01-18T23:35:28-08:00",
          "tree_id": "4a48be5fc8f26d41e0761741c8ef9a383d8dda93",
          "url": "https://github.com/zpzim/SCAMP/commit/0f8c54a43e8af41588172cf8696e793a43175f5b"
        },
        "date": 1642578241949,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 2120418039.9999928,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 8440512399.999989,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 33846802300.000034,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 1590015909.9999995,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 6327991300.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 26892950799.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1441046310.0000014,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 5691399599.999954,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 22763088100.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 1087055559.9999988,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 4269484300.000045,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 17066001699.9999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1542351700.0000062,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6318793000.000028,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24452368400.00007,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1168806989.999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4590205099.999934,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18391968800.000088,
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
          "id": "47244aab789c9741befde47dab6f997a922da9fa",
          "message": "Prevent accidental generation additional gpu threads when using pyscamp (#83)",
          "timestamp": "2022-02-06T14:51:09-08:00",
          "tree_id": "007b1ae0150f4fdbb4732fba0922c7bc626b8777",
          "url": "https://github.com/zpzim/SCAMP/commit/47244aab789c9741befde47dab6f997a922da9fa"
        },
        "date": 1644188427281,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 2571685099.999968,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 10142390599.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 40217843600.00015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 1890008830.0000107,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7812500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 7433410800.000047,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 30107993200.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1808387109.9999897,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 7190620100.00016,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 28591809899.999817,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 1334595070.0000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 5282235599.999922,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 21000448300.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1736653000.0000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7812500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6964052599.999832,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 27860841899.99997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1313035760.0000024,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 10937500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 5190703200.000144,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 20469505400.00012,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
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
          "id": "7ae38cee736de8182d2ee941f247d6067bb7fee9",
          "message": "Add cpu matrix reduction (#77)\n\n* Make matrix summaries more accurate by allowing a floating point to represent the width of an output cell.\r\n\r\n* Add test support for matrix summary profile types.\r\n\r\n* Allow run_tests.py to use pyscamp with matrix summaries\r\n\r\n* Update documentation to indicate that the matrix summary profile type is available when using CPUs only.\r\n\r\n* Add matrix summary pyscamp test.\r\n\r\n* Do not output a whitespace on the end of matrix summary output rows.",
          "timestamp": "2022-02-06T17:26:33-08:00",
          "tree_id": "04fa85f77e9a63dfba9833bbf91df2f87f21b85f",
          "url": "https://github.com/zpzim/SCAMP/commit/7ae38cee736de8182d2ee941f247d6067bb7fee9"
        },
        "date": 1644197663096,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1985443499.9999967,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 7893861799.999968,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 31161025099.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 1449813010.0000026,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 5850986400.0000105,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 22881791799.999973,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1319811149.999998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 5294998100.000044,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 20919051599.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 1000127899.9999955,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 3936812700.0000186,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 15843821199.999979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1380043580.000006,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5467358300.000001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21921480200.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1039936899.9999979,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4102343399.999995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 16473895399.999947,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}