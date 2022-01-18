# Contributing to SCAMP
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## We Actively Welcome Pull Requests
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've changed APIs, update the documentation.
3. Perform any necessary testing. (See Testing below)
4. Code will be automatically formatted by a github-actions commit.
5. Make a pull request!
6. Ensure the integration test suite which runs with the pull request passes.

## Testing
- Correctness issues should be caught by the SCAMP integration tests. These run on a pull request but you can also run them locally to verify before making a pull request:
  - To test general or CPU kernel specific changes run the following python script while in the `test/` directory: `python3 run_tests.py --executable "<path to SCAMP CLI binary>" --extra_args "--no_gpu --num_cpu_workers=2"`
  - To test GPU specific changes (required if you are making GPU specific changes), run the following python script while in the `test/` directory: `python3 ./run_tests.py --executable "<path to SCAMP CLI binary>"`
  - To test pyscamp changes, you can install the working version of pyscamp in your virtualenv using `python3 -m pip install .` and then run `python3 test/test_pyscamp.py`.
- **Important Note**: If you make CUDA changes, note that our integration test suite will not test the cuda-specific parts of your change. Please do a manual run on a GPU enabled system.

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/zpzim/SCAMP/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/zpzim/SCAMP/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Include input which is causing problems if possible!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

* 2 spaces for indentation rather than tabs
* C++/CUDA code will be automatically formatted on push by github-actions.

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)
