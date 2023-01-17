#!/bin/bash

set -e

echo "Do you wish to use a very low number of parameter configurations and epochs to quickly establish that the code is working? (y/n)"
read quick_str

if [[ $quick_str == "y" ||  $quick_str == "Y" ]]; then
    quick=--quick
    ML=""
elif [[ $quick_str == "n" ||  $quick_str == "N" ]]; then
    quick=""
    ML=--ML
else
    echo "Please re-run and type y or n"
    exit 1
fi

for model in NMVM GaussianProcess/nuFixed GaussianProcess/nuVaried
do
    echo ""
    echo "######## Starting simulation study for $model model ############"
    echo ""

    if [[ $model == "NMVM" ]]; then
        trainingm="[1,10,30,75,150,300]"
        m=150
    elif [[ $model == "GaussianProcess/nuVaried" ]]; then
        trainingm="[1,10,30,75,150]"
        m=30
    else
        trainingm="[1,10,30,75,150]"
        m=1
    fi

    ## Visualise field realisations
    julia --threads=auto --project=. src/Realisations.jl --model=$model
    # Rscript src/Realisations.R --model=$model # TODO

    ## Train the neural estimators
    julia --threads=auto --project=. src/Train.jl --model=$model $quick --m=$trainingm

    ## Estimation
    julia --threads=auto --project=. src/Assess.jl --model=$model --m=$m $ML

    ## Results
    Rscript src/$model/Results.R
done
