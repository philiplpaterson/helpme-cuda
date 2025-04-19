fullexample_parallel-aimos: fullexample_parallel.cpp fullexample_parallel.cu
	mpicxx -O3 fullexample_parallel.cpp -c -o fullexample_parallel-xlc.o -DHAVE_MPI=1 -DHAVE_FFTWD=1
	nvcc -O3 -arch=sm_70 fullexample_parallel.cu -c -o fullexample_parallel-nvcc.o -DHAVE_MPI=1 -DHAVE_FFTWD=1 -I/opt/ibm/spectrum_mpi/include
	nvcc -O3 -arch=sm_70 pain.cu -c -o pain.o -DHAVE_MPI=1 -DHAVE_FFTWD=1 -I/opt/ibm/spectrum_mpi/include
	mpicxx -O3 fullexample_parallel-xlc.o fullexample_parallel-nvcc.o pain.o -o fullexample_parallel.out -lcudadevrt -lcudart -lstdc++ -lcurand -lfftw3 -lm -DHAVE_MPI=1 -DHAVE_FFTWD=1 -L/usr/local/cuda-11.2/lib64/

fullexample_parallel-local: fullexample_parallel.cpp fullexample_parallel.cu
	mpicxx fullexample_parallel.cpp -c -o fullexample_parallel-mpicxx.o -DHAVE_MPI=1 -DHAVE_FFTWD=1 -L /usr/include/fftw3.h
	nvcc fullexample_parallel.cu -c -o fullexample_parallel-nvcc.o -DHAVE_MPI=1 -DHAVE_FFTWD=1 -I/usr/lib/x86_64-linux-gnu/openmpi/include
	nvcc -O3 pain.cu -c -o pain.o -DHAVE_MPI=1 -DHAVE_FFTWD=1
	mpicxx -O3 fullexample_parallel-mpicxx.o  fullexample_parallel-nvcc.o pain.o -o fullexample_parallel.out -L/usr/local/cuda/lib64/  -lcudadevrt -lcudart -lstdc++ -lcurand -lfftw3 -lm -lcusolver -DHAVE_MPI=1 -DHAVE_FFTWD=1
