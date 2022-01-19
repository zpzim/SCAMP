window.BENCHMARK_DATA = {
  "lastUpdate": 1642573407123,
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
      }
    ]
  }
}