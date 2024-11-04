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

echo "Do you wish to skip training? (y/n)"
read skip_training_str
if ! [[ $skip_training_str == "y" ||  $skip_training_str == "Y" || $skip_training_str == "n" ||  $skip_training_str == "N" ]]; then
    echo "Please re-run and type y or n"
    exit 1
fi

source sh/simulationstudies.sh     # Section 3
source sh/application.sh           # Section 4 
# source sh/supplement.sh            # Supplementary material 


echo ""
echo "######## Everything finished! ############"
echo ""
