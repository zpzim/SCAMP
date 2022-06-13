window.BENCHMARK_DATA = {
  "lastUpdate": 1655082205060,
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
        "date": 1654585701961,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3892508883.9996533,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 79848084.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2944572918.999256,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 67361268 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7966603769.000358,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66962042 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5774244797.999927,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66658824.00000001 ns\nthreads: 1"
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
        "date": 1654646070121,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3526963504.002197,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 72328068.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2939518523.9991407,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 63947292.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7961846466.991119,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66311289.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5771872874.00011,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64225683.00000003 ns\nthreads: 1"
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
        "date": 1654668140372,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3436695409.0073705,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 70662773 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2936987153.996597,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 62257041 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7959337720.9967,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64612199.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5769470180.006465,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 61757342.000000045 ns\nthreads: 1"
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
        "date": 1654791446686,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3493338650.9947014,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 72699878.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2939204561.000224,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64043529.999999985 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7960689706.989797,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66713600.00000004 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5771359995.9970455,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 63876170.00000001 ns\nthreads: 1"
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
        "date": 1655079885981,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3478116240.9926763,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 72957246 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2938547299.0106792,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 64306699.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7960908456.996549,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66568358.99999999 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5771458715.025802,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 63760560.00000002 ns\nthreads: 1"
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
        "date": 1655082193618,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_1NN_INDEX_SELF_JOIN/-1/1048576",
            "value": 3438374161.021784,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 72834491 ns\nthreads: 1"
          },
          {
            "name": "BM_1NN_SELF_JOIN/-1/1048576",
            "value": 2938638278.981671,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 63972336.000000015 ns\nthreads: 1"
          },
          {
            "name": "BM_SUM_SELF_JOIN/-1/1048576",
            "value": 7961725763.976574,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 66646693.00000001 ns\nthreads: 1"
          },
          {
            "name": "BM_MATRIX_SELF_JOIN/-1/1048576",
            "value": 5771189212.973695,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 63884942.99999997 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}