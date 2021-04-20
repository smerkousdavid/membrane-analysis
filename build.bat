@echo off
echo building cython extension
python setup.py build_ext --inplace
echo done