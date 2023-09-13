from os import environ
from distutils.unixccompiler import UnixCCompiler
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class build(build_ext):
    def build_extensions(self):
        if isinstance(self.compiler, UnixCCompiler):
            if 'zig' in self.compiler.cc:
                self.compiler.dll_libraries = []
                self.compiler.set_executable(
                    'compiler_so',
                    f'{self.compiler.cc} -O3 -Wall'
                )
            for ext in self.extensions:
                ext.undef_macros = ['_DEBUG']
        super().build_extensions()

if "--use-cython" in sys.argv:
    sys.argv.remove("--use-cython")
    use_cython = True
else:
    use_cython = False
include_dirs = environ['INCLUDE'].split(';')
suffix = "pyx" if use_cython else "c"
exts = [
    Extension(
        name='pynn.core',
        sources=['src\\pynn\\core.' + suffix],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pynn.gradfunc',
        sources=['src\\pynn\\gradfunc.' + suffix],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pynn.math',
        sources=['src\\pynn\\math.' + suffix],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pynn.loss',
        sources=['src\\pynn\\loss.' + suffix],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pynn.nn',
        sources=['src\\pynn\\nn.' + suffix],
        include_dirs=include_dirs,
    )
]
if use_cython:
    from Cython.Build import cythonize
    exts = cythonize(exts)
setup(
    ext_modules=exts,
    cmdclass={'build_ext': build},
)
