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

for model in GP/nuSigmaFixed GP/nuFixed Schlather
do

    echo ""
    echo "##### Starting simulation study for $model model #####"
    echo ""

    if [[ $model == "BrownResnick"  ]]; then
        bash sh/compile.sh
    fi

    if [[ $model == "GP/nuSigmaFixed" || $model == "GP/nuFixed" ]]; then
        m="[1]"
    elif [[ $model == "SPDE" ]]; then
        m="[1]"
    else
        m="[1,50]"
    fi

    # TODO prompt user if they wish to skip training (remember that is has to be runnable from all.sh)
    julia --threads=auto --project=. src/simulationstudy.jl --model=$model $quick --m=$m # --skip_training

    Rscript src/simulationstudy.R --model=$model
done
