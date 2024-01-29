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

# echo ""
# echo "##### Starting supplementary experiment S1: variable number of spatial locations #####"
# echo ""
# julia --threads=auto --project=. src/supplement/variablesamplesize.jl $quick
# Rscript src/supplement/variablesamplesize.R
# 
# echo ""
# echo "##### Starting supplementary experiment S2: neighbourhood selection #####"
# echo ""
# julia --threads=auto --project=. src/supplement/neighbourhood.jl $quick
# Rscript src/supplement/neighbourhood.R

echo ""
echo "##### Starting supplementary experiment S2: disc radius sensitivity #####"
echo ""
julia --threads=auto --project=. src/supplement/discradius.jl $quick
Rscript src/supplement/discradius.R