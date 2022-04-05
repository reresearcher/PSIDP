#!/usr/bin/env bash
python run.py --train --dataset 'flickr25k' --code-length 32 --num-query 2000 --num-train 5000
sleep 300 
python run.py --train --dataset 'flickr25k' --code-length 64 --num-query 2000 --num-train 5000
sleep 300 
python run.py --train --dataset 'flickr25k' --code-length 128 --num-query 2000 --num-train 5000
sleep 300 
python run.py --train --dataset 'flickr25k' --code-length 16 --num-query 2000 --num-train 5000