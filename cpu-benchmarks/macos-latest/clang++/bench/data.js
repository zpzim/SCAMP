window.BENCHMARK_DATA = {
  "lastUpdate": 1644783539286,
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
        "date": 1642479763671,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1063924804.5000045,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1898300 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4344896142.999971,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6869999.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 16867260508.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12595000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 779219400.7000034,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2099900.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3420138895.0000362,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7210000.000000008 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 13397798249.999994,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14808000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 632612766.2999966,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2208299.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2544994051.0000033,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6625999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10024942544.999989,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13165000.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 510974046.1000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2047499.9999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 2884408184.999984,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7424000.000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 11734347886.999956,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14240000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 2088711070.2999961,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2082800.0000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 8390771886.999971,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7389000.0000000065 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 34216883918.999996,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13809999.999999989 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1974486306.7000039,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2107299.9999999953 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 5332824968.000011,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7688000.000000028 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18977306017.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15469999.999999983 ns\nthreads: 1"
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
        "date": 1642528475841,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1602135198.6000013,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2078600 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 5826196429.999982,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7753999.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 27120018050.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14067999.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 925195579.8999974,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2190500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3480890185.9999537,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8530999.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 13487192320.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14979000.000000006 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 724360171.8999969,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2202799.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2932017940.9999695,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7098999.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10963942750.000002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13312000.000000019 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 478195184.00000334,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2130999.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 2061575930.8000007,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4176699.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 7739908957.000011,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 18911999.999999985 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1657618206.6000001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2304700.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6493040587.999986,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8382000.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25426856755.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16579000.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1206233253.5999985,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2416399.9999999963 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4779169035.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7853999.999999972 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18874042944.000053,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16216000.000000007 ns\nthreads: 1"
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
        "date": 1642565290631,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1123064563.1000016,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1958899.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4486578224.000141,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7114000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 18106128590.999786,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13491999.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 894238746.0999953,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2102200 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3628955234.9999213,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8018999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 13883443810.000017,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15677999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 674992520.6999934,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2043299.9999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2643370971.999957,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7352000.000000025 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10760086075.000118,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14666000.000000013 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 513223214.9999936,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2255599.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 2531340445.000069,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12446000.000000013 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 8518302595.000023,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16053000.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1706061371.6999795,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2253199.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6569945616.000041,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7348999.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 26019520662.000103,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13742000.000000032 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1213600357.80001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2198600.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 5744406395.000169,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7753999.999999983 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 21131674871.000088,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16048000.000000007 ns\nthreads: 1"
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
        "date": 1642566318697,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1473066195.8999972,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2089499.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 7024886360.999971,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6969000.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 25583677462.999958,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13510000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 801814285.1999983,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2199700.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3245513852.000045,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7574000.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 18745894151.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 19773999.999999985 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 699152605.5999998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2123200 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 3671056926.0000057,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6657999.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 12087347326.999975,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12758999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 484021316.4000033,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2105300.0000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 3298119096.0000505,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6947000.000000008 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 9662317538.000025,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14706999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 2153469159.6000013,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2083999.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 11099296652.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7317000.000000018 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 36342198510.99999,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14892999.999999989 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1628712215.6000009,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2333699.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 6026554528.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8064999.999999989 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 28226896116.000034,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15153000.000000028 ns\nthreads: 1"
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
        "date": 1642566662733,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1070187460.2999992,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2133399.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4329650837.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6775999.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 17459521487.00002,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13311999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 795582531.499997,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2086999.9999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3266588230.000025,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7188999.999999987 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 12551739085.000008,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14993999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 648754410.3000004,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2221700.0000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2579086028.9999957,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7088999.999999984 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10408659088.000036,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13729999.999999965 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 548448991.4999983,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2082299.9999999981 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 2078583051.1999962,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4484300.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 7879058752.000049,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 18923999.99999997 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1680065521.0999992,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2383499.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6497432776.999971,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7507000.000000041 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 26119547272.000034,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14315000.000000022 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1292663220.2000008,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2451299.9999999953 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4932371913.999987,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9023000.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 20232162238,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17818000 ns\nthreads: 1"
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
        "date": 1642573404334,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1755739322.2999961,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 6833900.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 6481711277.999921,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 28252000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 28157430947.999954,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 72486000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 1276770062.4999862,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 14755199.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 5636017605.000007,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 78476999.99999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 19704673797.000168,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 360626000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 1306294311.4000063,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 28458999.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 4729044228.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 48060999.99999991 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 17978666307.00014,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 79070000.0000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 745536300.2000012,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 18842200.000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 3058942651.999814,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 57960000.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 11720634254.000061,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 104363000.00000021 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 2062960260,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 8801199.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 11134327855.000038,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 36697000.0000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 32770166374.999916,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 121838999.9999998 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1372588407.1999872,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 9752599.999999978 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 5780563050.999945,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 35601000.00000066 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 21871631232.99979,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 106437000.00000012 ns\nthreads: 1"
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
        "date": 1642578119159,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1059412095.1999686,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1833599.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4225567537.999723,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6412000.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 16878462225.000021,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12840999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 778027779.3000095,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1971299.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3092563460.999827,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6994000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 12319338443.000107,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14056000.000000013 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 635151866.9999678,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1808999.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2518818635.000116,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6016999.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10070626401.000027,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12722000.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 464464934.0999877,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2029900.0000000012 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1819968340.100013,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3480600 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 7263260326.999898,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13323000.00000003 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1550132046.0999978,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1830300.0000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6218952236.99994,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6609000.000000032 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 24738787658.00009,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12963999.999999976 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1144112447.7999892,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2071000.0000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4637706160.000107,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7249999.999999979 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 18486086135.999813,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15160000.000000007 ns\nthreads: 1"
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
        "date": 1644188412763,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 1090842959.6000018,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2100699.9999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 4296387926.999955,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6779999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 17287266993.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13119999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 820878372.300001,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2421899.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 3269670713.999972,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8404999.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 13287032433.000036,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14303999.999999983 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 660729157.7999945,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1952700.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2614547654.9999104,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8474000.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10230471566.999994,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 11954999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 467807336.4999932,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2076499.9999999977 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1872455904.1000018,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3787900.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 7420869400.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14099000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1572375388.799992,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1939900 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 6317395605.999991,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6798000.000000026 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 25193108834.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15073999.999999976 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 1170143244.1000066,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2048699.9999999977 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 4820566924.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 8015999.999999967 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 19089164100.00006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13988000 ns\nthreads: 1"
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
        "date": 1644197565640,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 956999883.3000028,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1938600 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3823257899.999987,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7056999.999999997 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15159328284.999958,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13725000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 713249483.5999978,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2055400.0000000012 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2803505017.0000205,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7385000.000000017 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 11250306365.000029,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14905000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 629219693.4999993,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2220400.0000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2486704313.900003,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3678700.0000000014 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9641303489.999985,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15983999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 439750695.7000019,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2196499.9999999986 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1745691937.4999986,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3984800.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6846518854.00001,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17832999.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1280007689.8999975,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2398200.0000000005 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 5292535466.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7916999.999999952 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 20584031514.000004,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14067000.000000052 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 955563699.4999987,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2214399.999999994 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3905799753.999986,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7876999.999999967 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 15186667004.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15158999.999999978 ns\nthreads: 1"
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
        "date": 1644709015073,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 945243283.6999832,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1927000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3776299862.999622,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6897000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15017761175.000486,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13513999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 702586707.2999972,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2032500.0000000012 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2798869780.9996667,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7493000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 11072872377.999374,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15537000.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 598171644.9000487,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2183399.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2386876468.599985,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3812699.999999999 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9544236310.000088,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13280999.999999987 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 432638475.80001307,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2025300.0000000023 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1716500516.3999922,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3784199.9999999986 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6803464149.0003195,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14728000.000000019 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1109405337.8999887,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1996999.9999999988 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 4471762831.999513,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6870999.999999905 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 17922206590.999847,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15021000.000000007 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 842655210.2000642,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2938200.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3392221754.9998093,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 10475000.000000011 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 13526383370.99961,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 22511000.000000004 ns\nthreads: 1"
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
        "date": 1644722111350,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/32768",
            "value": 931146351.5000013,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 1877500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/65536",
            "value": 3732998368.000153,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6824999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 14848684875.99988,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13025000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/32768",
            "value": 695191198.0999739,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2365800 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/65536",
            "value": 2765613193.000263,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7204000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/2/131072",
            "value": 10958280773.999832,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14151999.999999998 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/32768",
            "value": 591070682.4000045,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2062399.9999999988 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/65536",
            "value": 2415404090.3999887,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 4158500.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 9687626511.999952,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15571000.000000002 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/32768",
            "value": 440279404.20000964,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2181500 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/65536",
            "value": 1741851044.59998,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 3843900 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/2/131072",
            "value": 6958856808.999826,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14645999.999999993 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/32768",
            "value": 1132318915.6999888,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2245699.999999995 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/65536",
            "value": 4560779400.9997635,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6828000.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 18175775524.000072,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 14172999.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/32768",
            "value": 840849986.4999611,
            "unit": "ns/iter",
            "extra": "iterations: 10\ncpu: 2115399.9999999953 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/65536",
            "value": 3366461878.9998713,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7898000.000000016 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/2/131072",
            "value": 13465513907.00006,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 15270000.000000006 ns\nthreads: 1"
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
        "date": 1644780262134,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 23691487678.999977,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 13671999.999999996 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 13777965504.999998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7924000.000000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 23540500107.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7350000.000000003 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 52060800087.000015,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 7799000 ns\nthreads: 1"
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
        "date": 1644783538295,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/1/131072",
            "value": 15609290565.000038,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 12941000 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/1/131072",
            "value": 10021110206.999992,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6539000 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/1/131072",
            "value": 18682290862.999935,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 6893000.000000004 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/1/131072",
            "value": 43559332822.99998,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 10333999.999999996 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}