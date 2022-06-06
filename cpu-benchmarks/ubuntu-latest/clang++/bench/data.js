window.BENCHMARK_DATA = {
  "lastUpdate": 1654540580839,
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
        "date": 1642566561939,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 817270428.2000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3187910 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3274665184.999975,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6688699.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13009582416.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13023400.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 592705327.9000006,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2941410.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2358780249,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7373240 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9427961721.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13908599.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 381558121.80000055,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1587720.0000000035 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1510743011.1000013,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6246170.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6021549192.000009,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12345099.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 270805998.9999981,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2810219.9999999967 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1061446137.6000007,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7147009.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4225841692.999978,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13807899.999999983 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1320185017.900002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1627179.9999999949 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5290818399.999978,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5437499.999999984 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21170917799.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12573999.999999974 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 974979192.7999979,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2933020.0000000084 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3913738599.999988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6909700.000000019 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15625001170.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14005700.00000001 ns\nthreads: 1"
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
        "date": 1642572219467,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 849052697.0000019,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3255840 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3404122611.9999804,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6983700.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13618068492.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13520600.000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 621242150.4000019,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3047360.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2474683836.600002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7640200 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9929280675.000029,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15078999.999999981 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 445486441.3999985,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1679199.9999999974 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1762447512.7999973,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6549949.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 7073585901.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13159499.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 320672517.99999213,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2969100.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1249204373.8999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7352569.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4928574785.000024,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14423799.999999987 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1343763201.1999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1706289.9999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5399182134.999932,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5660299.999999952 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21591453708.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13178200.000000028 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 993594949.4999931,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3066000.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3987837842.9999657,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7616399.999999967 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15968643570.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14684699.999999994 ns\nthreads: 1"
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
        "date": 1642578035538,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 984157673.8000015,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4738370 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3913642185.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7687100.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15644567383.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15026200.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 712118223.4999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3868500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2844474453.9999933,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8110199.999999984 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 11359649757.999989,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16746199.999999989 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 459268570.8000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1909220.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1833806658.8999991,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 8349810.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 7295753219.999994,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14771099.999999981 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 325411985.4999999,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3321269.9999999986 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1274240045.9999998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 9142499.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5032961018.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16137699.999999922 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1585153528.5000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1728340.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6340306166.000005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6420599.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25420354263.99997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14732300.000000032 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1167649471.8000016,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3990190.0000000047 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4678897011.999993,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7883100.000000032 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18845924647.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17060300.00000003 ns\nthreads: 1"
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
        "date": 1644188206902,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1196069165.999998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3613080.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4671949704.000013,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7555199.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 18991161436.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 19756800.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 814668391.1999986,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3451700.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3205312157.0000086,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8079000.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 12785840076.999989,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16508499.999999981 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 626478793.1000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1982850.0000000026 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2488704322.3000007,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6849140.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10066110678.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14843700.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 411197595.90000343,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3338479.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1620160273.9999998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7888650.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6331458126.999962,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13950100.00000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1475762477.699999,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2034510.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5889512905.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6643799.999999978 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 23493273878.999958,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12741100.000000061 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1073993487.1000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3583500.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4337490347.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8928099.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 17213365317.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16405499.99999996 ns\nthreads: 1"
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
        "date": 1644197477927,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 707006110.9999983,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3292290.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 2828021946.0000067,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6998499.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 11268487128.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13359499.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 513816057.70000076,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2974970.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2041219207.5000007,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7370900 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 8134466309.999993,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14146899.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 376791588.5999997,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1645849.9999999986 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1483469574.5000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6340699.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5902271810.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12586500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 270805644.09999967,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2928720.0000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1047639889.6999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7110109.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 4112356345.9999843,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14116800.000000041 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1313453733.8000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1642009.9999999937 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5260564306.999982,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5638700.000000107 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21056190129.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12793399.999999955 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 967470230.3999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2869339.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3877164816.0000043,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7030800.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15518896001.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14286299.999999974 ns\nthreads: 1"
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
        "date": 1644708835510,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 712494990.4000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3216160.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 2861005808.0000157,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6879300.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 11274452915.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13380299.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 527727695.50000477,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2936379.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2088967601.999997,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7512390.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 8345742964.0000105,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14442800.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 343094783.80000085,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1626389.9999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1350626498.700001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6374069.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5316725532.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12727800.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 243118008.51999977,
            "unit": "ns/iter",
            "extra": "iterations: 100\ncpu: 2884183 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 938431271.6999943,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7140630.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 3669880846.000069,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14250199.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1185666355.6000056,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1609269.9999999958 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 4763567920.000014,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5507699.999999893 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 19064778993.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12920400.000000054 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 885651462.6000035,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2959390.0000000065 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3537899684.000081,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7249899.99999992 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 14080535176.000011,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14414499.999999996 ns\nthreads: 1"
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
        "date": 1644721925128,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 706856684.4999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3073270 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 2844783456.000016,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6768100.0000000065 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 11390728938.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 18551500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 532571973.09999925,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2910250.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2068496220.2000008,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7412510 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 8162291375.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14180800.00000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 337638642.79999894,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1535019.9999999981 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1345831170.9000015,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6327969.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5324520367.000019,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12241000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 244365815.5100002,
            "unit": "ns/iter",
            "extra": "iterations: 100\ncpu: 2854087.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 926986980.8000009,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7149280.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 3630135592.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13803800.000000034 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1183168697.0000021,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1580470.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 4757327434.000047,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5556300.00000007 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 19165789085.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13110100.000000013 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 875051214.399997,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2861219.9999999977 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3477488236.0000563,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7268300.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 13901560447.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13967899.999999922 ns\nthreads: 1"
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
        "date": 1644780251778,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13620683536.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16426900.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 8777780777.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13524700 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 17911533395.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13548199.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 52128820682,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15347700.000000006 ns\nthreads: 1"
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
        "date": 1644783149009,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13067915532,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15149799.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6212750706.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13666500.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22067529809.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13938500.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 46468905586.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13044500 ns\nthreads: 1"
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
        "date": 1644793488446,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 29035784955.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16356600.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6180220259.999999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13995700 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 23307185072.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14077300 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 49405619237.00005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13817600 ns\nthreads: 1"
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
        "date": 1644827484081,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 12848532133.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15598400.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5689560268.999997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13085000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21753890910.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12941800.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 46390048404,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13278600.000000002 ns\nthreads: 1"
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
        "date": 1645899205191,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 17379487441.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 18369100 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10576482693.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14064600.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22296304188.000023,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15473899.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 50973578126.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14563699.999999998 ns\nthreads: 1"
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
        "date": 1654484677392,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13476842992.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8999000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 8227607974.000079,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7815799.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 17322820144.999924,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8025399.999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 34852174131.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7837300.000000006 ns\nthreads: 1"
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
        "date": 1654540579722,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 14461589605,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13902100 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6903898446.999989,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11838000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21075535681.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12101300.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 43096038732.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12105600.000000007 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}