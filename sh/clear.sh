#!/bin/bash

set -e

echo "Really clear all intermediates, figures, and results? (y/n)"
read answer

if [[ $answer == "y" || $answer == "Y" ]]; then
    # rm -rf intermediates/*
    # rm -rf img/*

    # delete all files in the intermediates directory except for .gitignore 
    find intermediates/* ! -name ".gitignore" -type f -exec rm -f {} +

    # delete all files in the img directory except for .gitignore and schematic.png
    find img/* ! -name ".gitignore" ! -name "schematic.png" -type f -exec rm -f {} +
fi
