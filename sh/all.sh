#!/bin/bash
unset R_HOME

set -e

echo ""
echo "######## Setting up ############"
echo ""

echo "Do you wish to use a very low number of parameter configurations and epochs to quickly establish that the code is working? (y/n)"
read quick_str

if ! [[ $quick_str == "y" ||  $quick_str == "Y" || $quick_str == "n" ||  $quick_str == "N" ]]; then
    echo "Please re-run and type y or n"
    exit 1
fi

Rscript src/pointprocesses.R       # Point-process figure in Section 2
source sh/simulationstudies.sh     # Section 3
source sh/application.sh           # Section 4
source sh/supplement.sh            # Supplementary material

echo ""
echo "######## Everything finished! ############"
echo ""
