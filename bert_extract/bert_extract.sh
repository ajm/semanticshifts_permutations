#!/bin/bash

module purge
module load python-env/2019.3
unset PYTHONPATH
source ../env/bin/activate

date

echo
echo

python bert_extract.py $1 $2 $3 $4 $5

echo
echo

date

