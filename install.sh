#!/usr/bin/env bash

rm -Rf .conda

conda create            \
  --yes                 \
  --no-default-packages \
  --prefix .conda       \
  python=3.7            \
  requests              \
  numpy                 \
  pandas                \
  matplotlib            \
  seaborn               \
  bokeh
