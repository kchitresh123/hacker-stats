#!/usr/bin/env bash

rm -Rf .conda

conda env create -f environment.yml -p .conda
