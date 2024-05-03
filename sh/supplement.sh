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

#TODO add option to skip training

echo ""
echo "##### Starting supplementary experiment S1: neighbourhood definitions #####"
echo ""
Rscript src/supplement/S1neighbourhood_definitions.R
echo ""
echo "##### Starting supplementary experiment S1: neighbourhood selection #####"
echo ""
julia --threads=auto --project=. src/supplement/S1neighbourhood.jl $quick
Rscript src/supplement/S1neighbourhood.R

echo ""
echo "##### Starting supplementary experiment S2: prior measure for S and variable number of spatial locations #####"
echo ""
julia --threads=auto --project=. src/supplement/S2variablesamplesize.jl $quick
echo ""
echo "##### Starting supplementary experiment S2: prior measure for S and simulation efficiency #####"
echo ""
julia --threads=auto --project=. src/supplement/S2simulationefficiency.jl $quick
Rscript src/supplement/S2.R
