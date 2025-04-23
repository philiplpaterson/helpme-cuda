#!/bin/bash
sbatch -N 1 --partition=dcs-2024 --gres=gpu:1 -t 60 ./run1.sh 1 1 512 waterbox526848 175
sbatch -N 1 --partition=dcs-2024 --gres=gpu:2 -t 60 ./run1.sh 8 2 512 waterbox526848 175
sbatch -N 2 --partition=dcs-2024 --gres=gpu:3 -t 60 ./run1.sh 27 3 512 waterbox526848 175
sbatch -N 2 --partition=dcs-2024 --gres=gpu:4 -t 60 ./run1.sh 64 4 512 waterbox526848 175
sbatch -N 4 --partition=dcs-2024 --gres=gpu:4 -t 60 ./run1.sh 125 5 512 waterbox526848 175