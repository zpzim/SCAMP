import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
          out = subprocess.check_output(['cmake', '--version'])
        except OSError:
          raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < LooseVersion('3.15.0'):
          raise RuntimeError("CMake >= 3.15.0 is required")

        try:
            out = subprocess.check_output(['nvcc', '--version'])
        except OSError:
          print('WARNING: CUDA was not found on the system, to build with CUDA, verify nvcc can be found in the PATH')
 
        if sys.maxsize <= 2**32:
          print('WARNING: building/using pyscamp on a 32 bit platform is unsupported.')

        for ext in self.extensions:
          self.build_extension(ext)

    def build_extension(self, ext):
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        python_exe = os.environ.get('PYSCAMP_PYTHON_EXECUTABLE_PATH', sys.executable)
        cmake_args = ['-DPYTHON_EXECUTABLE=' + python_exe]
        cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]

        force_cuda = os.environ.get("FORCE_CUDA", "")
        force_no_cuda = os.environ.get("FORCE_NO_CUDA", "")

        # This environment variable is a way to opt out of platform auto-selection on windows.
        # It may be useful if build errors occur on windows related to setting CMAKE_GENERATOR_PLATFORM.
        do_not_auto_select_cmake_platform = os.environ.get("PYSCAMP_NO_PLATFORM_AUTOSELECT", "")

        # Default to release build.
        build_type = os.environ.get("PYSCAMP_BUILD_TYPE", "Release")
        # We need to set CMAKE_BUILD_TYPE here in case we aren't using a multi-config generator (e.g. Ninja)
        env['CMAKE_BUILD_TYPE'] = build_type

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]

        # Build pyscamp module.
        cmake_args += ["-DBUILD_PYTHON_MODULE=TRUE"]

        if force_cuda:
          cmake_args += ["-DFORCE_CUDA={}".format(force_cuda)]

        if force_no_cuda:
          cmake_args += ["-DFORCE_NO_CUDA={}".format(force_no_cuda)]

        if platform.system() == "Windows":
          # Make sure the libraries get placed in the extdir on Windows VS builds.
          cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(build_type.upper(), extdir)]
          # On older versions of Visual Studio, we may need to specify the generator platform manually
          # as it defaults to 32-bit compilation. Note that we can set this here regardless of the
          # generator used because SCAMP's CMakeLists.txt will remove the setting if it is unused.
          cmake_generator_platform = os.environ.get("CMAKE_GENERATOR_PLATFORM", "")
          if not cmake_generator_platform and sys.maxsize > 2**32 and not do_not_auto_select_cmake_platform:
            env['CMAKE_GENERATOR_PLATFORM'] = 'x64'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        configure_cmd = ['cmake', ext.sourcedir] + cmake_args
        print("Configuring SCAMP")
        print(' '.join(configure_cmd))
        subprocess.check_call(configure_cmd, cwd=self.build_temp, env=env)

        build_cmd = ['cmake', '--build', '.', '--target', ext.name, '--config', build_type, '--parallel', '4']
        print("Building SCAMP")
        print(' '.join(build_cmd))
        subprocess.check_call(build_cmd, cwd=self.build_temp)

setup(
    name='pyscamp',
    use_scm_version = {
        "root": ".",
        "relative_to": __file__,
        "local_scheme": "no-local-version"
    },
    setup_requires=['setuptools_scm'],
    author='Zachary Zimmerman',
    author_email='zpzimmerman@gmail.com',
    description='SCAlable Matrix Profile',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('pyscamp')],
    packages=find_packages(),
    cmdclass=dict(build_ext=CMakeBuild),
    url="https://github.com/zpzim/SCAMP",
    zip_safe=False
)
