#!/bin/bash

set -e

echo ""
echo "Compiling C code..."
echo ""

touch src/BrownResnick/PairwiseLikelihoodBR.c # touch the script to force recompilation
R CMD SHLIB src/BrownResnick/PairwiseLikelihoodBR.c

echo ""
echo "Finished compiling C code."
echo ""
