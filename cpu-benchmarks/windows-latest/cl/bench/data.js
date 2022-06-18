window.BENCHMARK_DATA = {
  "lastUpdate": 1655513056573,
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
          "id": "e1335e2c753c67ee1d1777118eb7a1e05379607c",
          "message": "Use Eigen to improve maintainability and cross-platform performance of CPU kernels. (#86)\n\n* Refactored CPU Kernels to use Eigen.\r\n\r\n* Revert Windows CUDA build to windows-2019 due to Action incompatibility with windows-2022",
          "timestamp": "2022-02-12T15:28:55-08:00",
          "tree_id": "7e187e3230cb419e471a7261867093c03326a774",
          "url": "https://github.com/zpzim/SCAMP/commit/e1335e2c753c67ee1d1777118eb7a1e05379607c"
        },
        "date": 1644709438711,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 4230801300.0001664,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 16841775599.999891,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 67371875000.000046,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 3257260299.999871,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 12727391799.99985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 50796396799.99991,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 3743771800.000104,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 15297654599.999987,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 61524933500.00009,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 2801558499.9998283,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 11423525199.999859,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 45391938200.00014,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 2072777659.999997,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 8347937800.000181,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 32844140900.000183,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1555350060.0000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1562500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 6147049400.000014,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 24396977300.00012,
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
          "id": "2b94acd64d074207debd201a682d82a378ce61b5",
          "message": "Fix pyscamp sdist to include the appropriate files to build the distribution. (#90)\n\nAlso modifies tests to verify the sdist can be built from the archive before a release.",
          "timestamp": "2022-02-12T19:07:07-08:00",
          "tree_id": "6837c153f8b3fdfcd5fe3be3c0acd46b443c59c2",
          "url": "https://github.com/zpzim/SCAMP/commit/2b94acd64d074207debd201a682d82a378ce61b5"
        },
        "date": 1644722657155,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 3360474599.9998612,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 13923575499.99997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 53664625900.000145,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 2411944220.0000095,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3125000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 9798553399.999947,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 39288898000.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 2928221099.9999733,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 11785629099.999825,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 47267389399.999954,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 2161686310.000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 8775964100.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 35142138199.99986,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1606770810.000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4687500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6311360899.999954,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25145951400.000057,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1187296909.9999864,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4759340599.99986,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18897649799.999954,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
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
          "id": "a09aed826cc4867f4754138678270bb8475afedd",
          "message": "Refactor benchmarks suite, add gpu benchmarks. (#88)\n\n* Refactor Benchmark library and add GPU benchmarks\r\n\r\n* Reduce the number of CPU benchmarks computed.",
          "timestamp": "2022-02-13T11:17:05-08:00",
          "tree_id": "7059afba734693fece8c8570a08d8b1c0aced0f4",
          "url": "https://github.com/zpzim/SCAMP/commit/a09aed826cc4867f4754138678270bb8475afedd"
        },
        "date": 1644780740452,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 66562227000.00012,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 62137342300,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 31971169799.99987,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 96323576300.00002,
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
          "id": "98b089b14c3f51de0a729419e586d52c04c716b9",
          "message": "Add various improvements to compile options. Including Automatic AVX detection for MSVC. (#91)\n\n* Add various improvements to compile options. Adds automatic AVX detection for MSVC.",
          "timestamp": "2022-02-13T11:40:41-08:00",
          "tree_id": "7fa6d66af821da480935020a421cf503470dc817",
          "url": "https://github.com/zpzim/SCAMP/commit/98b089b14c3f51de0a729419e586d52c04c716b9"
        },
        "date": 1644783938617,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 45612419499.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 26781894899.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25917464800.00006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 62869254800.00014,
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
          "id": "3e7d3b8cb92cd7c09ac2233e46dc0fcd2e9ef5ac",
          "message": "Remove 'fast' reduction path for CPU kernels and go with the Eigen path. (#92)\n\nIt appears to almost always be faster.",
          "timestamp": "2022-02-13T15:00:45-08:00",
          "tree_id": "696158b8b6b58f0fb83e433729609c07c054397c",
          "url": "https://github.com/zpzim/SCAMP/commit/3e7d3b8cb92cd7c09ac2233e46dc0fcd2e9ef5ac"
        },
        "date": 1644794009871,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 23149029799.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 19768017299.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24020840599.999928,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 56077001800.00006,
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
          "id": "4db3e253c1ed87927cec18f7e2ec7bb138dc418a",
          "message": "Apply optimizations to CPU Kernel reductions to improve performance on various toolchains. (#93)\n\n* Optimize CPU Kernel Reductions for Various compilers.",
          "timestamp": "2022-02-14T00:21:30-08:00",
          "tree_id": "a1624d0c1d5d3440e4103798610d734eeb050301",
          "url": "https://github.com/zpzim/SCAMP/commit/4db3e253c1ed87927cec18f7e2ec7bb138dc418a"
        },
        "date": 1644828338450,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 30676157999.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 26093346200.000042,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 28561508299.999786,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 64677155300.00009,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
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
          "id": "8d40eb8bb81038df0a4a11277148511c5d2f1404",
          "message": "Clean up pyscamp build logic in setup.py (#96)\n\n* Stop trying to detect visual studio in setup.py.\r\n* Increases the required cmake version for pyscamp to be 3.15 or more.\r\n* Move autoselection of CMAKE_GENERATOR_PLATFORM to inside CMakeLists.txt\r\n* Pass CMAKE_GENERATOR_PLATFORM on Windows when compiling pyscamp. Remove it in SCAMP's CMakeLists.txt if it is not needed.\r\n* Change recommendations for how to specify environment variables to set compilers/generators for SCAMP and pyscamp, This should be more aligned with the normal usage of cmake.\r\n* Update documentation.",
          "timestamp": "2022-02-26T10:08:58-08:00",
          "tree_id": "4aae813ae8b1a4130dd6da85648e908b4ed2670b",
          "url": "https://github.com/zpzim/SCAMP/commit/8d40eb8bb81038df0a4a11277148511c5d2f1404"
        },
        "date": 1645899748318,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 29329481299.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 23350710500.000103,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 27856258999.99991,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 68820145200.00008,
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
          "id": "ae79670ead75043a8b246b6626f49a69b54fa8e0",
          "message": "Allow SCAMP binaries to be optionally redistributable (#99)\n\n* Added runtime dispatch of AVX/AVX2-based CPU kernels. These are conditionally compiled only if they are needed to produce a redistributable binary.\r\n\r\n* Add option to disable -march=native configurations and make the SCAMP binary redistributable. This is specified via the environment variable SCAMP_ENABLE_BINARY_DISTRIBUTION=ON\r\n\r\n* Adds some flags to increase the chance a compiler will use FMA instructions when they are available.\r\n\r\n* Add testing coverage for redistributable binary builds. Including emulation tests with Intel SDE to verify SIMD dispatch runs on various CPU configurations.\r\n\r\n* Update main CMakeLists.txt to better specify global compile flags for different build types.\r\n\r\n* Update docker container to build in a redistributable way. \r\n\r\n* Update CUDA build tests to use updated action to build on windows-latest.\r\n\r\n* Minor performance tuning of CPU kernel unroll widths.\r\n\r\n* Prevent unnecessary files from being packaged with pyscamp",
          "timestamp": "2022-06-05T19:59:41-07:00",
          "tree_id": "09d6ffee8145b7738ee7142f89546e754abacc37",
          "url": "https://github.com/zpzim/SCAMP/commit/ae79670ead75043a8b246b6626f49a69b54fa8e0"
        },
        "date": 1654485095489,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 29205196500.000057,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 22847051800.000145,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 28539187399.999947,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 67571937200.00009,
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
          "id": "392c2e43c99f6807d8fb3d05b94577671e33cafb",
          "message": "Fix linker errors that can occur when building pyscamp shared libs. (#100)",
          "timestamp": "2022-06-06T11:31:28-07:00",
          "tree_id": "201dfd18c8c58da88dfd61c69a49928907937be8",
          "url": "https://github.com/zpzim/SCAMP/commit/392c2e43c99f6807d8fb3d05b94577671e33cafb"
        },
        "date": 1654541183513,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 31439479000.00012,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 26157144599.99981,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 28437641500.000153,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 64522786600.000015,
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
          "id": "83c8e32c9e78f64f3f3936d999b054d43530a46e",
          "message": "Add verbosity to pyscamp builds in CI tests.",
          "timestamp": "2022-06-06T14:54:36-07:00",
          "tree_id": "5343818d68962f10d3a7f37abf4bfcddd776e137",
          "url": "https://github.com/zpzim/SCAMP/commit/83c8e32c9e78f64f3f3936d999b054d43530a46e"
        },
        "date": 1654554347600,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 30974833799.999943,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 23053619600.000046,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 23202300499.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 62059043900.00001,
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
          "id": "4cc2208c7f620b796f83c81d5a8cb62e22e2c277",
          "message": "Fix some build issues with redistributable binaries on windows and mac (#102)\n\n* Disable clang-tidy when building by default for now.\r\n\r\nThere are some issues with newer versions of clang-tidy which can cause broken builds.\r\n\r\n* Only depend on cpu_features when building a redistributable binary.\r\n\r\nMove Eigen dependency into the cpu_kernels module.\r\n\r\n* Add message during configuration when clang-tidy isn't enabled.\r\n\r\n* Add Action to publish dev packages to test pypi.",
          "timestamp": "2022-06-07T00:05:03-07:00",
          "tree_id": "b17c0d4ab66ca4954c3e83e058cc110e0aa5729a",
          "url": "https://github.com/zpzim/SCAMP/commit/4cc2208c7f620b796f83c81d5a8cb62e22e2c277"
        },
        "date": 1654587348488,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 22310249200.000042,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 18729084899.999863,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22244938499.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 55561294099.99994,
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
          "id": "47ab9235124f2d03961064473505491af513a8dd",
          "message": "Clean up python module (#103)\n\n* Enable publishing intermetiate versions of SCAMP between releases to test pypi.\r\n\r\n* Only build the SCAMP executable if we aren't building the python module.\r\n\r\n* Only include relevant source files in pyscamp.\r\n\r\n* Bump pybind11 to v2.9.2\r\n\r\n* Add verbose messages indicating how cmake is invoked by pyscamp's setup.py\r\n\r\n* Fix failure detection in architecture emulation test.\r\n\r\n* Reduce warning spam from MSVC. Stop using -Wall on MSVC use W4 instead.\r\n\r\n* Fix pyscamp build issues that occur when CMAKE_BUILD_TYPE is set incorrectly and we aren't using a multi-config generator. Should fix some issues with using Ninja.\r\n\r\n* Allow a custom Python executable path to be provided when building pyscamp.",
          "timestamp": "2022-06-07T16:51:15-07:00",
          "tree_id": "09aef9b0ef24eb0bd5600b39cc6068cb4dce7df5",
          "url": "https://github.com/zpzim/SCAMP/commit/47ab9235124f2d03961064473505491af513a8dd"
        },
        "date": 1654646867451,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 21633348099.99992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 17635637800.000042,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22222029700.000122,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 52987094899.99988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
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
          "id": "f6e4f20d1781534c80df4e6bcf0faa19d02b53aa",
          "message": "Sickbay windows hardware emulation test for now.",
          "timestamp": "2022-06-07T22:58:16-07:00",
          "tree_id": "fda452409f2b59289e9d32316a3395f83fe6e7a8",
          "url": "https://github.com/zpzim/SCAMP/commit/f6e4f20d1781534c80df4e6bcf0faa19d02b53aa"
        },
        "date": 1654668757971,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 32253662800.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 26717261299.99979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 29004119100.000027,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 66213573900.00003,
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
          "id": "82b2d4329be76cf9089dbbd7847b3260558c4c2d",
          "message": "Cleanup to support conda packaging of pyscamp (#105)\n\n* Prevent SCAMP from using non-MPL-licenced code from EIGEN.\r\n\r\n* Add option to build pyscamp with external installs of pybind11 and eigen\r\n\r\n* Moves CMake dependencies closer to where they are used in the project.\r\n\r\n* Only configure the python module when building pyscamp.",
          "timestamp": "2022-06-09T09:14:37-07:00",
          "tree_id": "a5c9ef494ccf75d05f0d5780534a5daaa63852f8",
          "url": "https://github.com/zpzim/SCAMP/commit/82b2d4329be76cf9089dbbd7847b3260558c4c2d"
        },
        "date": 1654791773866,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 22170823600.00006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 18620920899.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22287812899.99995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 55404311399.999985,
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
          "id": "b40a3f419d7eeb0b39c11cdaf6fd250fe7de1b86",
          "message": "Update SCAMP to require CUDA 11.0 or greater. Update CUDA builds to use updated CMake Modules. (#109)\n\n* Add the ability to specify arbitrary arguments to cmake from the environment during pyscamp setup.\r\n\r\n* Add explicit support for GeForce Ampere and GeForce Maxwell GPUs.\r\n\r\n* Use CMAKE_CUDA_ARCHITECTURES. Remove deprecated usage of FindCUDA and use FindCUDAToolkit instead.\r\n\r\n* Avoid using check_language if we are requiring CUDA and instead directly enable the language. It is broken in some rare circumstances.\r\n\r\n* Drop support for CUDA versions less than 11. Update CMake minimum requirement to 3.18.\r\n\r\n* Dockerfile now installs new version of cmake via pip.\r\n\r\n* Remove multiply linked libcudart. Just link it once in libcommon if needed.",
          "timestamp": "2022-06-12T17:21:08-07:00",
          "tree_id": "2f270dc98d71b4bf12f2eaa98c3646de6854aee2",
          "url": "https://github.com/zpzim/SCAMP/commit/b40a3f419d7eeb0b39c11cdaf6fd250fe7de1b86"
        },
        "date": 1655080333323,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 27003278899.999943,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 22532857199.99985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 28001247599.999943,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 67139951899.999916,
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
          "id": "c1ed842d5ba957c62c77d08dc820fdf594964feb",
          "message": "Update docs with information on conda package availability (#111)\n\n* Update pyscamp docs with info about conda package.\r\n\r\n* Update build instructions for windows CUDA builds. Add disclaimer that 32-bit builds are unsupported and non x86_64 configurations are untested.\r\n\r\n* Add pointer to automated GPU benchmarks in documentation.",
          "timestamp": "2022-06-12T17:59:38-07:00",
          "tree_id": "5268a66d5aaa797f8c3c5b677244a896cd82395d",
          "url": "https://github.com/zpzim/SCAMP/commit/c1ed842d5ba957c62c77d08dc820fdf594964feb"
        },
        "date": 1655082521482,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 27544826399.99992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 23541359800.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 31250000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24236631799.999943,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 64326347299.999954,
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
          "id": "b29a2a96798b667e77eb1d8b110a866f9b1efa9e",
          "message": "Update requirements.txt to match SCAMP required cmake.",
          "timestamp": "2022-06-12T19:08:52-07:00",
          "tree_id": "bebc057bdd167a764a50a57152cbb13506917d55",
          "url": "https://github.com/zpzim/SCAMP/commit/b29a2a96798b667e77eb1d8b110a866f9b1efa9e"
        },
        "date": 1655086820163,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 20373310599.99997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 17173263399.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21019153899.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 51838438499.99994,
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
          "id": "005ffa35e66d46167f07beded35af58a54056b7e",
          "message": "Add issue templates (#114)",
          "timestamp": "2022-06-13T15:56:48-07:00",
          "tree_id": "f472c1452e8311795f945d34caee23de15a7c892",
          "url": "https://github.com/zpzim/SCAMP/commit/005ffa35e66d46167f07beded35af58a54056b7e"
        },
        "date": 1655162800497,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 22248889500.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 18622857200,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22290851500.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 55360536600.000046,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 0 ns\nthreads: 1"
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
          "id": "aa6f74794ddaad1472c5ddaea92d0bcf3bdb6884",
          "message": "Update documentation with additional information. (#117)\n\n* Added additional information on the pyscamp conda packages throughout the documentation.\r\n* Added performance comparisons to other matrix profile libraries.",
          "timestamp": "2022-06-17T10:11:58-07:00",
          "tree_id": "2d465fcdc25ae93e7a77a9961c46a99d29147067",
          "url": "https://github.com/zpzim/SCAMP/commit/aa6f74794ddaad1472c5ddaea92d0bcf3bdb6884"
        },
        "date": 1655487670606,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 23959597600.000053,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 19909928499.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25142246399.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 64074414800.000114,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15625000 ns\nthreads: 1"
          }
        ]
      },
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
        "date": 1655513052168,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 2.066269980000004,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0046875 s\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1.6939570300000013,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.0078125 s\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1.8138233599999922,
            "unit": "s/iter",
            "extra": "iterations: 10\ncpu: 0.00625 s\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/32768",
            "value": 4.200629100000015,
            "unit": "s/iter",
            "extra": "iterations: 1\ncpu: 0.015625 s\nthreads: 1"
          }
        ]
      }
    ]
  }
}