from os import environ
from distutils.unixccompiler import UnixCCompiler

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


class build(build_ext):
    def build_extensions(self):
        if isinstance(self.compiler, UnixCCompiler):
            if 'zig' in self.compiler.cc:
                self.compiler.dll_libraries = []
                self.compiler.set_executables
                self.compiler.set_executable(
                    'compiler_so',
                    f'{self.compiler.cc} -O3 -Wall'
                )
            for ext in self.extensions:
                ext.undef_macros = ['_DEBUG']
        super().build_extensions()


include_dirs = environ['INCLUDE'].split(';')
exts = (
    Extension(
        name='pynn.core',
        sources=['src\\pynn\\core.pyx'],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pynn.gradfunc',
        sources=['src\\pynn\\gradfunc.pyx'],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pynn.math',
        sources=['src\\pynn\\math.pyx'],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pynn.loss',
        sources=['src\\pynn\\loss.pyx'],
        include_dirs=include_dirs,
    )
)
setup(
    ext_modules=cythonize(exts, language_level=3),
    zip_safe=False,
    package_dir={'pynn': 'src\\pynn'},
    cmdclass={'build_ext': build},
)
