from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

import numpy as np
import os

# name the module
MODULE = 'analysis'
LANG_LEVEL = 3

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
    name='cell-structure',
    version='1.0',
    description='Najafian Lab Image Cell Structure Analysis. A structural analysis, of membranes and other cellular component, based on electron microscopy images with complex datasets',
    author='David Smerkous',
    author_email='smerkd@uw.edu',
    include_package_data=True,
    packages=find_packages(include=['structure', 'structure.*']),
    install_requires=[
        'numpy>=1.14.5',
        'opencv-python>=4.3.0.36',
        'scikit-image>=0.18.1',
        'cython>=0.29.0',
        'htmlmin>=0.1.12',
        'jsmin>=2.2.2',
        'rcssmin>=1.0.6',
        'wxpython>=4.1.1',
        'imutils>=0.5.0',
        'imagecodecs',
        'six'
    ],
    ext_modules=cythonize(
        [make_ext(ext) for ext in scandir('.')],
        language_level=LANG_LEVEL,
        # nthreads=1# ,
        # gdb_debug=True
    )
)