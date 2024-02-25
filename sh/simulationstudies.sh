#!/bin/bash
unset R_HOME

set -e

if [ ! ${quick_str} ]; then
  echo "Do you wish to use a very low number of parameter configurations and epochs to quickly establish that the code is working? (y/n)"
  read quick_str
fi

if [[ $quick_str == "y" ||  $quick_str == "Y" ]]; then
    quick=--quick
elif [[ $quick_str == "n" ||  $quick_str == "N" ]]; then
    quick=""
else
    echo "Please re-run and type y or n"
    exit 1
fi

for model in GP/nuSigmaFixed GP/nuFixed # Schlather # BrownResnick
do

    echo ""
    echo "##### Starting simulation study for $model model #####"
    echo ""

    if [[ $model == "BrownResnick"  ]]; then
        bash sh/compile.sh
    fi

    if [[ $model == "GP/nuFixed" || $model == "GP/nuFixedSigmaVaried" ]]; then
        m="[1]"
    elif [[ $model == "SPDE" ]]; then
        m="[1]"
    else
        m="[1,20]"
    fi

    echo ""
    echo "##### Constructing and assessing point estimators #####"
    echo ""
    julia --threads=auto --project=. src/simulationstudy.jl --model=$model $quick --m=$m #--skip_training

    echo ""
    echo "##### Constructing and quantile estimators #####"
    echo ""
    julia --threads=auto --project=. src/simulationstudy-credibleinterval.jl --model=$model $quick --m=$m # --skip_training

    Rscript src/simulationstudy.R --model=$model
done
