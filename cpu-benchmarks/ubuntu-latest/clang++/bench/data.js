window.BENCHMARK_DATA = {
  "lastUpdate": 1642566141784,
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
          "id": "0b4569279a9a5715ddc3a09c665be57c74b5c16a",
          "message": "Fix",
          "timestamp": "2022-01-17T20:16:14-08:00",
          "tree_id": "c7c65bc016c756060509550bcd21775e4b10d94c",
          "url": "https://github.com/zpzim/SCAMP/commit/0b4569279a9a5715ddc3a09c665be57c74b5c16a"
        },
        "date": 1642479679894,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 816477385.8999994,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3272650.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3273260582.999995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7018700.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13067143369.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13669899.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 594752421.6999995,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2999830.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2374355589.6000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7406890 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9480431835.999979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14526100.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 383245417.79999775,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1639069.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1517193878.9999986,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6394930.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6035835625.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12703499.99999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 271845439.0999995,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2935399.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1063147794.5999991,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7221339.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4194917435.999997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13989500.000000099 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1321136891.5000007,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1614969.9999999932 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5295290399.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5474900.000000061 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21184114789.999966,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13023200.000000013 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 973351200.7000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2941590.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3897435800.000039,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6990599.999999958 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15620520353.99995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14323000.000000086 ns\nthreads: 1"
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
          "id": "2733b2e210219e9558c127d54968a1d0e650298e",
          "message": "Update cpu-benchmarks.yml",
          "timestamp": "2022-01-18T09:47:44-08:00",
          "tree_id": "39e0c303d31223fbbad8167debd6a99be4ddf55c",
          "url": "https://github.com/zpzim/SCAMP/commit/2733b2e210219e9558c127d54968a1d0e650298e"
        },
        "date": 1642528351061,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 815164008.9000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3292630.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3255935811.000029,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6812699.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13010319207.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13063500.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 593461114.0000015,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2984069.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2365252446.2999964,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7473130 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9456017837.999979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14302599.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 382285249.50000066,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1601039.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1514927488.5000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6359030.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6007277907.999991,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12538300.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 271168681.59999967,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2857910.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1062573925.7000021,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7295439.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4201505147.999967,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14010300.000000032 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1319701728.0000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1631499.9999999967 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5285098785.000003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5498800.000000026 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21157766284.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12847400.000000065 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 973601490.6000036,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3003640.0000000075 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3897146294.0000377,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7394200.000000018 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15606920553.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14468900.000000007 ns\nthreads: 1"
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
          "id": "72274eb699cbb066fb3b638c09404d47213ab159",
          "message": "Fix benchmark output path on Windows. Also compile benchmarks with clang-cl on Windows.",
          "timestamp": "2022-01-18T20:00:24-08:00",
          "tree_id": "3ded6f12eef7525e3758420a8cfbf5d2a1bbbaaf",
          "url": "https://github.com/zpzim/SCAMP/commit/72274eb699cbb066fb3b638c09404d47213ab159"
        },
        "date": 1642565165776,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 930007404.0000026,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4796340 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3735606345.999997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7637600.000000008 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 14843977055.000038,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15004900.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 675339852.6000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4565750.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2705447077.0000305,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7599200 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 10791720479.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16608199.999999989 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 436846004.10000485,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2542130.0000000037 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1726132757.8000019,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7567809.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6849125986.999979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13998600.000000028 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 312064365.20000124,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4394780.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1231364255.599999,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 9355940.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4768015878.000029,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15869299.999999976 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1499309221.9999995,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2502219.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6023775572.999967,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6189599.999999906 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24117065061.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14774300.000000019 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1113181790.3000011,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4245600.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4444830305.999971,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7883500.000000043 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 17725078505,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 19290999.999999948 ns\nthreads: 1"
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
        "date": 1642566141032,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 816050664.6000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3192840.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3280399329.9999943,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6782500.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13051176950.000013,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13300300.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 595132688.7000022,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2954530.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2359518246.4000006,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7449580.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9468911457.999979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14160900.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 383740219.0000006,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1649539.9999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1517578756.8000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6362860.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6045482888.999999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12358199.999999985 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 271663660.3000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2903450.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1057955135.0000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7279520 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4201426977.0000114,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14220200.000000017 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1320108956.6999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1633930.0000000056 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5295191406.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5521800.000000021 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21182418060.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12708399.999999953 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 973107891.799998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2968999.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3908642026.9999847,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7063399.999999942 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15636846353.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14430300.000000007 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}