window.BENCHMARK_DATA = {
  "lastUpdate": 1654552761874,
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
        "date": 1644827069007,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3428735576.104373,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 71640073 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2939002492.9307404,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62837767.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7926826410.926878,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 65526906.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5770686085.801572,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62763673.99999999 ns\nthreads: 1"
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
        "date": 1645899191559,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3437662672.0186324,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 71027782.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2936989593.959879,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62768858.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7960083596.990444,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64792583.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5769513359.002303,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62166590.00000002 ns\nthreads: 1"
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
        "date": 1654484602245,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3574698430.951685,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 72844387.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 3045144028.9616585,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64087477.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 8255882341.880351,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66789124.99999998 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5982052383.013069,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 63898169.00000001 ns\nthreads: 1"
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
        "date": 1654540495186,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3568158294.0742373,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 73751663 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 3048313431.907445,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64887439.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 8257551052.607595,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 67755806.00000003 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5983201299.794018,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64283032.99999999 ns\nthreads: 1"
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
        "date": 1654552750501,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3574103535.1529717,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 73922145 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 3049409267.026931,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 65573046.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 8256254583.131522,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 67826452.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5987172814.08608,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 65187283.00000004 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}