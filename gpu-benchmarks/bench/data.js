window.BENCHMARK_DATA = {
  "lastUpdate": 1644793569603,
  "repoUrl": "https://github.com/zpzim/SCAMP",
  "entries": {
    "Benchmark": [
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
        "date": 1644780139179,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3436079255.770892,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 71211620 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2938054548.110813,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62837513.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7926748527.213931,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 65405683.00000002 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5770451938.267797,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62616296.999999985 ns\nthreads: 1"
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
        "date": 1644784377845,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3433724745.1767325,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 74037310 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2943486778.996885,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 65017337.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7931927475.14695,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 67874505.00000003 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5774162684.101611,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64769753.999999955 ns\nthreads: 1"
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
        "date": 1644793557885,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3478062147.2746134,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 70729308 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2940256823.785603,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64213896.00000002 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7928797485.77252,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66503232.00000002 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5770040878.094732,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62029115.99999999 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}