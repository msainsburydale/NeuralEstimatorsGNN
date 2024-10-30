#!/bin/bash
unset R_HOME

set -e

# Prompt user if they wish to do a "quick" run to test the code
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


# Prompt user if they wish to skip training 
if [ ! ${skip_training_str} ]; then
  echo "Do you wish to skip training? (y/n)"
  read skip_training_str
fi

if [[ $skip_training_str == "y" ||  $skip_training_str == "Y" ]]; then
    skip_training=--skip_training
elif [[ $skip_training_str == "n" ||  $skip_training_str == "N" ]]; then
    skip_training=""
else
    echo "Please re-run and type y or n"
    exit 1
fi

for model in GP/nuSigmaFixed GP/nuFixed Schlather   
do

    echo ""
    echo "##### Starting simulation study for $model model #####"
    echo ""

    if [[ $model == "Schlather" ]]; then
        m="[1,20]"
    else
        m="[1]"
    fi

    julia --threads=auto --project=. src/simulationstudy.jl --model=$model $quick $skip_training --m=$m 
    Rscript src/simulationstudy.R --model=$model
done
