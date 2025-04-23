#!/bin/bash
module load xl_r spectrum-mpi cuda/11.2
mpirun --bind-to core --map-by ppr:32:node -np $1 ./fullexample_parallel_mpic.out $2 $2 $2 $3 $4 $5