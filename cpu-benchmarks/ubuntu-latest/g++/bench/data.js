window.BENCHMARK_DATA = {
  "lastUpdate": 1642572234885,
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
        "date": 1642479694134,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 956256053.3000009,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3470220 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3910230748.999993,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8368399.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15458748420.000006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14909400.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 657026226.7000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3171749.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2667976860.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7196400.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 10409627900.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14836200.000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 655445185.7000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2009800.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2586804182.0000086,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6493899.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10492855531.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13266300.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 433591822.3999983,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3254340 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1692877505.0000014,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7532330.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6765364120.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14181699.999999965 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1602736335.7000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1970420.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6527929248.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5978499.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25629927813.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12946999.999999987 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1217401865.2999961,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3503689.9999999986 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4855665439.000007,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8711799.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 19122393775.999966,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14998300.00000002 ns\nthreads: 1"
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
        "date": 1642528407335,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1000091341.000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4930100.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4006631294.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9251700.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15949622556.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15575300 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 728453353.9999983,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4333990 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2908766872.9999905,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 10270399.999999985 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 11600372448.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17531500.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 607467601.7999991,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2308910 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2422918361.8999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 8388969.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9675432276.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14620200.000000028 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 435571608.5999972,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4084890.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1727394471.7999997,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 8947650.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6858347894.999951,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16643999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1688155852.5000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2346289.9999999953 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6761712767.999995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7114100.000000012 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 27070125725.000027,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 19245699.99999992 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1235692983.0999944,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4385340.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4964101256.999982,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9552899.999999976 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 19933281116.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17370500.000000067 ns\nthreads: 1"
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
        "date": 1642565173442,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 768081538.4000027,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3216510.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3346130148.000043,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6968099.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13394838013.000027,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13342699.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 554767217.9000018,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2970389.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2213793335.3999987,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7276550.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 8939677258.99995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14594099.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 471931877.10000193,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1673020 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2028599229.1000016,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6693800.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 8103510026.000037,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12651300.000000032 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 365293767.2000007,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2994780.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1443260421.299999,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7302260.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5729363610.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14281900.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1412296450.3999982,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1823469.9999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5666710136.999995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5891399.999999991 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22680102865.00004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14325599.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 982189442.8999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2935210.0000000047 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3801997077.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7127800.000000018 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15202332182.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14875199.999999978 ns\nthreads: 1"
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
        "date": 1642566169141,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 838917180.9000003,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3206010 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3357046149.000013,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11788899.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13396905294.999983,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13541899.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 609491693.1,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2992280 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2421897622.0999994,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7546880 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9684038033.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14960900 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 509617427.39999926,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1577299.9999999981 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2025616737.9999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6389340.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 8087524559.999991,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12837700.000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 365024560.5000009,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2958440.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1449954774.1000016,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7356700 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5723864354,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15006199.99999997 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1411932798.800001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1598680.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5665603055.999952,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6234399.999999973 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22681222876.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13360800.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1038246389.8000027,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2977929.9999999953 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4158177756.0000434,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7193200.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 16680670371.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14675900.000000075 ns\nthreads: 1"
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
        "date": 1642566622742,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 940207288.9000009,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3675680.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3746024379.000005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 10003000.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 14902413338.999964,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16024300.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 658372872.6000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3575130 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2606653056.000027,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9438400 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 10287970342.999983,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17784500.000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 627405771.2,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2400130.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2495551309.900003,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7465910 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9937462656.99997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15530500.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 427224544.49999535,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3537430.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1670872563.1999984,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 8448509.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6686756692.999949,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17126499.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1554686491.4999957,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2373419.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6193444157.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6821799.999999989 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25034849383.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16214000.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1135496435.5999982,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3868780.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4571558223.0000105,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9001999.999999955 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18225583174.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17449600.000000063 ns\nthreads: 1"
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
        "date": 1642572234100,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 868415806.5000019,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3372350 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3472917283.999948,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7574000.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13874566105.999975,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13647600.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 632414768.8000039,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2926680.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2517467586.9999986,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6968700.000000008 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 10247690040.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14621199.999999987 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 616300745.2999977,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1660549.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2440831205.200004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6363720.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9469971740.000006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12399899.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 431768614.39999926,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2861280.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1700541341.1000006,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7364690.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6747338204.000016,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14530099.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1595673481,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1661760.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6402019822,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5572999.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25623057832.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15312199.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1172108973.5999954,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2942770 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4705816366.999954,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7073999.999999969 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18893766824.999943,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14643299.99999997 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}