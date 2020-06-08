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
        if cmake_version < LooseVersion('3.12.0'):
            raise RuntimeError("CMake >= 3.12.0 is required")

        cmake_help = ''
        try:
            cmake_help = subprocess.check_output(['cmake', '--help'])
        except OSError:
            raise RuntimeError("Cmake could not be queried for its default generator")

        # Check if visual studio is the default cmake generator
        if '* Visual Studio' in cmake_help.decode():
          self.cmake_vs_default_generator = True
        else:
          self.cmake_vs_default_generator = False

        try:
            out = subprocess.check_output(['nvcc', '--version'])
        except OSError:
           print('CUDA was not found on the system, to build with CUDA, verify nvcc can be found in the PATH')
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        force_cuda = os.environ.get("FORCE_CUDA", "")
        force_no_cuda = os.environ.get("FORCE_NO_CUDA", "")
        cmake_cpp_compiler = os.environ.get("CMAKE_CXX_COMPILER", "")
        cmake_cuda_compiler = os.environ.get("CMAKE_CUDA_COMPILER", "")
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")     
        cmake_toolset = os.environ.get("CMAKE_GENERATOR_TOOLSET", "")        

        build_type = os.environ.get("BUILD_TYPE", "Release")
        build_args = ['--config', build_type]

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]
        cmake_args += ["-DBUILD_PYTHON_MODULE=TRUE"]

        if force_cuda:
          cmake_args += ["-DFORCE_CUDA={}".format(force_cuda)]

        if force_no_cuda:
          cmake_args += ["-DFORCE_NO_CUDA={}".format(force_no_cuda)]

        if cmake_cpp_compiler:
          cmake_args += ["-DCMAKE_CXX_COMPILER={}".format(cmake_cpp_compiler)]

        if cmake_cuda_compiler:
          cmake_args += ["-DCMAKE_CUDA_COMPILER={}".format(cmake_cuda_compiler)]      

        if cmake_generator:
          cmake_args += ["-DCMAKE_GENERATOR={}".format(cmake_generator)]

        if cmake_toolset:
          cmake_args += ["-DCMAKE_GENERATOR_TOOLSET={}".format(cmake_toolset)]
          


        if platform.system() == "Windows":
            generator_is_vs = False
            # If the user specified a visual studio generator OR the default generator is visual studio
            # then we need to specify the correct options for the visual studio generator
            if 'Visual Studio' in cmake_generator:
              generator_is_vs = True
            elif not cmake_generator and self.cmake_vs_default_generator:
              generator_is_vs = True

            if generator_is_vs:
              cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(build_type.upper(), extdir)]
              build_args += ['--', '/m']
            else:
              cmake_args += ['-DCMAKE_BUILD_TYPE=' + build_type]
              build_args += ['--', '-j4']
          
            if sys.maxsize > 2**32 and generator_is_vs:
                cmake_args += ['-A', 'x64']

        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + build_type]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake',
                               '--build', '.',
                               '--target', ext.name
                               ] + build_args,
                              cwd=self.build_temp)

setup(
    name='pyscamp',
    version='0.4.1',
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
