# ReadMe for running cuda and MPI for fullexample_parallel

To compile with cuda using your system of choice, need to include mpi when compiling nvcc for fullexample_parallel.cu. Find include using `mpicxx --showme:incdirs`

For my local setup with openmpi and cuda, you can see the Makefile for how to run.
