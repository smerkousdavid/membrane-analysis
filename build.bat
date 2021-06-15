@echo off
echo building cython extension
python setup.py build_ext --inplace --force
echo done