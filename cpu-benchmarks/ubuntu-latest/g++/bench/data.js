window.BENCHMARK_DATA = {
  "lastUpdate": 1655082234609,
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
        "date": 1642578040668,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 837764620.4000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3343860 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3350440166.999988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7558700.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13383986007.999994,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13585199.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 608898693.0000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3042950 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2423954783.4999995,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7599930 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9666054263.000006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14669800.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 509431558.2999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1674649.9999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2040895753.8000011,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6613280.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 8214083413.999987,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13289900.00000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 375360704.000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2965219.9999999963 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1442045510.3000023,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7402470 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5713980299.000013,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14625600.000000017 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1411490784.7999973,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1697359.9999999977 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5655715220.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6085699.999999972 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22680376553.99996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12944299.999999965 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1040155378.2999996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3013620.0000000056 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4158009004.000007,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7646399.999999942 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 16663807571.000006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14894100.00000002 ns\nthreads: 1"
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
        "date": 1644188154083,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 902728410.9000008,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3346140.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3638963345.0000076,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7249800.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 14509688458,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13693500.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 660099929.3000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3019049.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2658126108.000005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7458900.000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 10590459930.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14653299.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 507972136.3000005,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1648710.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2048244341.8000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6370509.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 8170227632.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12379900 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 363724085.3999998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2921650.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1308033811.9999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7178750 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5182572665.999999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14613600.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1399728006.800001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1674690.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5612395458.000009,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6128899.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22474379102.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12786499.999999978 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1026255114.8,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3041549.9999999986 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4125428841.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7794300.000000032 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 16536326395.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14946999.999999989 ns\nthreads: 1"
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
        "date": 1644197515865,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 868185333.8000025,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3216740 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3523006416.0000153,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6916899.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 14036780152.000006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13818299.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 638311434.5999985,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2929890.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2562380205.000011,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7056699.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 10102575826.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14313600.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 534608240.8000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1609919.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2054362378.9000008,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6343019.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 8181424481.999983,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12259700.000000013 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 374218966.5000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2872479.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1497195316.4000013,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7134079.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5818109794.9999695,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13997299.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1539773590.9999995,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1607899.9999999956 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6183778762.000031,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6026099.999999979 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24743739769,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14664500.000000024 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1131964438.6999984,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2856750.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4543855937.999979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6810199.999999989 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18220347446.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14074100.000000034 ns\nthreads: 1"
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
        "date": 1644708827876,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 814295803.5999982,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3257540.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3479812098.9999576,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7677500.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13456118597.000057,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14326999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 654548386.2000026,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3052830.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2504540874.999975,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7146100.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9466489789.99997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14254999.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 485504515.599996,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1644450 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1944163155.299998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6323460 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 7678974809.0000105,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 21709800 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 333306008.60000455,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2899030.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1379388278.9000008,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7204709.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5346937386.999969,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15164000.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1391006419.4999962,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1621400.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5595753064.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5959700.000000012 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22424726643.999973,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13122700.000000043 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1007735481.100002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2958690.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3958718982.999983,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7780000.000000009 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 16018620628.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15296499.99999999 ns\nthreads: 1"
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
        "date": 1644721933093,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 781730449.5000002,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3340970 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3122919058.999997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6935700.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 12407342846.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13520699.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 595478213.8999975,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3122550.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2372098772.6000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7699049.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 9316413861,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14007400.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 451664021.4000006,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1694709.9999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 1785662399.7000015,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6563690 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 7348570283.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12714000.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 326083427.0999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3024440.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1333456506.5000021,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 7321370 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 5275187424.000024,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13975100.000000019 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1324613489.2000043,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1659969.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5273087844.000031,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 5952899.999999983 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 21957119656.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13337400 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1023636343.8000011,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2995289.999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4284434230.9999776,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8107499.999999934 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 16276220770.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16598600.000000019 ns\nthreads: 1"
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
        "date": 1644780405031,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 18798181948.99997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16482100 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 12620669351.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14368500 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 28925840167.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14896399.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 50160445186.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14374999.999999998 ns\nthreads: 1"
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
        "date": 1644783194749,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13149070945.999994,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15015100 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 7601836047.999995,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13958499.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 23553752862.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13360100 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 83124511855.00003,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14063600.000000002 ns\nthreads: 1"
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
        "date": 1644793505172,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 24110510121.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13824700.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5590754157.000049,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11962699.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 23039199108.00005,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12348399.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 81565101169.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12188400.000000002 ns\nthreads: 1"
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
        "date": 1644827475492,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 18938497448.000023,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17059500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 11166307877.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 19889900.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 29012265573.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15603800.000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 50077367695.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14284000.000000006 ns\nthreads: 1"
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
        "date": 1645899199957,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 18697931232.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16991300 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 11076360566.000006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14141900.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 28668096055.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14375700.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 49996574123.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14276400.000000002 ns\nthreads: 1"
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
        "date": 1654484676708,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13466671109.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14316600.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5603369068.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12021899.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24122926186.000088,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12781200.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 50267202702.00008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12082700.000000002 ns\nthreads: 1"
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
        "date": 1654540606021,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 16015472913.999987,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13815500.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9307127972.999979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11807200.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25943836499.999973,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11978799.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 61083185286.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11922599.999999998 ns\nthreads: 1"
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
        "date": 1654554088340,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15488544681.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17555200 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6838876225.000035,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13719500.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 26618710764.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16294799.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 60598600674.00007,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13973399.999999996 ns\nthreads: 1"
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
        "date": 1654586594317,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 16067290474.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13140800.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9307958078.000013,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11285300.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25914374474.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11367000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 61024550948.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11193200 ns\nthreads: 1"
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
        "date": 1654646096221,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 12738867675.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14202200.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5598424674,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12264400.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22060383833.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12352400.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 50218016030,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12527100 ns\nthreads: 1"
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
        "date": 1654668200770,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15959417912.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13092000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9290338660.000032,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11264000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25923411382.999973,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11549900.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 61063249005.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11277500.000000004 ns\nthreads: 1"
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
        "date": 1654791525558,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 16061158635.000027,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13536300.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9315427295.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11716800 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25925665114.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11745100.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 61203974792.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11720500.000000002 ns\nthreads: 1"
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
        "date": 1655079895055,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13273177422.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14897700 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 6421586913.999989,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12700799.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 23342840432.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14189099.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 53356483836.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12682900.000000004 ns\nthreads: 1"
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
        "date": 1655082233806,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 13065094870.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14743899.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 5753514015.999997,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12274799.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 22466984679.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12566399.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 51261027160.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12314900.000000004 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}