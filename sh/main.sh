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

for model in GP/nuFixed Schlather BrownResnick # SPDE 
do

    echo ""
    echo "##### Starting main results for $model model #####"
    echo ""

    if [[ $model == "GP/nuFixed"  ]]; then
        m="[1]"
    elif [[ $model == "SPDE" ]]; then
        m="[1]"
    elif [[ $model == "BrownResnick" ]]; then
        m="[1,20]"
    else
        m="[1,20]"
    fi

    julia --threads=auto --project=. src/main.jl --model=$model $quick --m=$m # --skip_training

    if [[ $model == "GP/nuFixed"  ]]; then
        julia --threads=auto --project=. src/mainCI.jl --model=$model $quick --m=$m # --skip_training
    fi

    #Rscript src/main.R   --model=$model
done
