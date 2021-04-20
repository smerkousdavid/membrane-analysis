@echo off
echo building cython extension
python setup.py build_ext
copy build\lib.win-amd64-3.8\analysis\* analysis\
echo done