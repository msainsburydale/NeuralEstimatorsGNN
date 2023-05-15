#!/bin/bash

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

Rscript src/Setup.R

if [[ ! -f data/redseatemperature.rdata ]]
then
    echo "The Red Sea data set has not been downloaded, or is in the wrong location. Please see the README for download instructions."
fi

# Each .sh files asks the user if quick = y/n. To automate this
# script, we pipe the above response to each .sh file
yes $quick_str | bash sh/Misc.sh                 # Miscellaneous results
yes $quick_str | bash sh/theoretical.sh          # Section 2
yes $quick_str | bash sh/simulations.sh          # Section 3 and some sections of the Supplementary Material
yes $quick_str | bash sh/application.sh          # Section 4

echo ""
echo "######## Everything finished! ############"
echo ""
