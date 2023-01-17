#!/bin/bash

set -e

echo "Do you wish to use a very low number of parameter configurations and epochs to quickly establish that the code is working? (y/n)"
read quick_str

if [[ $quick_str == "y" ||  $quick_str == "Y" ]]; then
    quick=--quick
elif [[ $quick_str == "n" ||  $quick_str == "N" ]]; then
    quick=""
else
    echo "Please re-run and type y or n"
    exit 1
fi

for model in GaussianProcess/nuFixed GaussianProcess/nuVaried Schlather
do

    if [[ $model == "GaussianProcess/nuFixed" ]]; then
        m="[1]"
    else
        m="[1,30]"
    fi

    echo ""
    echo "######## Starting experiment 0 for $model model ############"
    echo ""
    julia --threads=auto --project=. src/experiment0.jl --model=$model $quick --m=$m

    echo ""
    echo "######## Starting experiments 1 and 2 for $model model ############"
    echo ""
    julia --threads=auto --project=. src/experiments.jl --model=$model $quick --m=$m

    Rscript src/Results.R --model=$model
done
