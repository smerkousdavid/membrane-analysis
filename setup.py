from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

import numpy as np
import os

# name the module
MODULE = 'analysis'

# special cases
SPECIAL = {
    'treesearch': {
        'compile': ['-std=c++14']  # add extra compile args to this file
    },
    'hitmiss': {
        'compile': ['-std=c++14']
    },
    'statistics': {
        'compile': ['-std=c++14']
    }
}


def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith('.pyx'):
            files.append(path.replace(os.path.sep, '.')[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def clean_path(path):
    if path.startswith('.'):
        return clean_path(path[1:])
    return path


def make_ext(ext):
    path = clean_path(ext).replace('.', os.path.sep) + '.pyx'
    name = MODULE + '.' + os.path.basename(clean_path(ext).replace('.', os.path.sep)).replace(os.path.sep, '')
    special = SPECIAL[name] if name in SPECIAL else {}
    return Extension(
        clean_path(ext) if special.get('name', None) is None else special.get('name', None),
        [path] + special.get('sources', []),
        include_dirs=['.', np.get_include()] + special.get('includes', []),
        extra_compile_args=['-O3', '-Wall'] + special.get('compile', []),
        extra_link_args=['-g'],
        libraries=[]
    )


# this is just for the cython files and is not designed to actually build the all-in-one application yet
setup(
    name='Membrane Analysis',
    packages=['analysis'],
    ext_modules=cythonize(
        [make_ext(ext) for ext in scandir('.')],
        language_level=3,
        # nthreads=1# ,
        # gdb_debug=True
    )
)