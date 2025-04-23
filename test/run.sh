#!/bin/bash
module load xl_r spectrum-mpi cuda/11.2
# $1 is the total number of ranks
# $2 2 2 are the nx ny nz
# $3 threads, which stays constant

# $4 grid size
# $5 file name
mpirun --bind-to core --map-by ppr:32:node -np $1 ./fullexample_parallel_mpic.out $2 $2 $2 $3 $4 $5