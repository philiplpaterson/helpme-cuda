#!/bin/bash
sbatch -N 1 --partition=el8-rpi --gres=gpu:1 -t 60 ./run.sh 1 1 512 waterbox12288 49
sbatch -N 1 --partition=el8-rpi --gres=gpu:2 -t 60 ./run.sh 8 2 512 waterbox41472 75
sbatch -N 2 --partition=el8-rpi --gres=gpu:3 -t 60 ./run.sh 27 3 512 waterbox139968 112
sbatch -N 2 --partition=el8-rpi --gres=gpu:4 -t 60 ./run.sh 64 4 512 waterbox255552 140
sbatch -N 4 --partition=el8-rpi --gres=gpu:4 -t 60 ./run.sh 125 5 512 waterbox526848 175