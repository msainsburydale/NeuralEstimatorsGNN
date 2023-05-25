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

for model in GP/nuFixed #GP/nuVaried Schlather
do

    echo ""
    echo "##### Starting experiments for $model model #####"
    echo ""

    if [[ $model == "GP/nuFixed" ]]; then
        m="[1]"
    else
        m="[1,30]"
    fi

    julia --threads=auto --project=. src/experiments/architectures.jl   --model=$model $quick --m=$m
    julia --threads=auto --project=. src/experiments/graphstructures.jl --model=$model $quick --m=$m --neighbours=radius
    julia --threads=auto --project=. src/experiments/graphstructures.jl --model=$model $quick --m=$m --neighbours=fixednum
    julia --threads=auto --project=. src/experiments/samplesize.jl      --model=$model $quick --m=$m --neighbours=radius
    julia --threads=auto --project=. src/experiments/samplesize.jl      --model=$model $quick --m=$m --neighbours=fixednum

    Rscript src/experiments/architectures.R   --model=$model
    Rscript src/experiments/graphstructures.R --model=$model

done
