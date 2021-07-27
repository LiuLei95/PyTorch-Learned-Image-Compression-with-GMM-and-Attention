#!/bin/bash

mkdir -p images

for i in {01..24..1}; do
  echo ${i}
  wget http://r0k.us/graphics/kodak/kodak/kodim${i}.png -O images/kodim${i}.png
done
