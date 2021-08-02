@echo off
echo building cython extension
call conda activate membrane-analysis
python setup.py build_ext --inplace --force
echo done