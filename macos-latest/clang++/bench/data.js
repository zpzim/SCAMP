window.BENCHMARK_DATA = {
  "lastUpdate": 1642565291740,
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
      }
    ]
  }
}