#!/bin/bash
echo "building membrane analysis extensions"
echo "please check CC/CXX vars (to point to anaconda gcc) if build fails with unrecognized file"
CONDA_ACTIVATE=$HOME/anaconda3/bin/activate
CONDA_ENV=membrane-analysis
conda activate $CONDA_ENV || echo using backup location && source $CONDA_ACTIVATE $CONDA_ENV
python setup.py build_ext --inplace
echo "done"