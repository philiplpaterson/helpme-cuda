# Not sure if the parallel aimos works
fulexample_parallel-aimos: fullexample_parallel_cuda.cpp fullexample_parallel_cuda.cu
	mpicxx -O3 fullexample_parallel_cuda.cpp -c -o fullexample_parallel_cuda-xlc.o -DHAVE_MPI=1 -DHAVE_FFTWD=1 -std=c++11 
	nvcc -O3 -arch=sm_70 fullexample_parallel_cuda.cu -c -o fullexample_parallel_cuda-nvcc.o -lcufft -lcufftw -DHAVE_MPI=1 -DHAVE_FFTWD=1 -I/opt/ibm/spectrum_mpi/include
	nvcc -O3 -arch=sm_70 pain.cu -c -o pain.o -DHAVE_MPI=1 -DHAVE_FFTWD=1 -I/opt/ibm/spectrum_mpi/include
	mpicxx -O3 fullexample_parallel_cuda-xlc.o fullexample_parallel_cuda-nvcc.o pain.o -o fullexample_parallel_cuda.out -lcudadevrt -lcudart -lstdc++ -lcurand -lcufft -lcufftw -lm -DHAVE_MPI=1 -DHAVE_FFTWD=1 -L/usr/local/cuda-11.2/lib64/ -std=c++11 

fullexample_parallel-local: fullexample_parallel_cuda.cpp fullexample_parallel_cuda.cu fullexample_parallel.cpp
	mpicxx fullexample_parallel_cuda.cpp -g -c -o fullexample_parallel_cuda-mpicxx.o -DHAVE_MPI=1 -DHAVE_FFTWD=1 -L /usr/include/fftw3.h
	nvcc fullexample_parallel_cuda.cu -g -c -o fullexample_parallel_cuda-nvcc.o -lcufft -lcufftw -DHAVE_MPI=1 -DHAVE_FFTWD=1 -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/local/cuda/lib64
	nvcc -O3 pain.cu -g -c -o pain.o -DHAVE_MPI=1 -DHAVE_FFTWD=1
	mpicxx -O3 fullexample_parallel_cuda-mpicxx.o  fullexample_parallel_cuda-nvcc.o pain.o -o fullexample_parallel_cuda.out -g -L/usr/local/cuda/lib64/  -lcudadevrt -lcudart -lstdc++ -lcurand -lm -lcusolver -lcufft -lcufftw -DHAVE_MPI=1 -DHAVE_FFTWD=1
	mpicxx fullexample_parallel.cpp -g -o fullexample_parallel_mpic.out -DHAVE_MPI=1 -DHAVE_FFTWD=1 -DBUILD_STANDALONE -L /usr/include/fftw3.h -lfftw3

fullexample_untouched: fullexample_parallel.cpp
	mpicxx fullexample_parallel.cpp -g -o fullexample_parallel_mpic1.out -DHAVE_MPI=1 -DHAVE_FFTWD=1 -DBUILD_STANDALONE  -lfftw3 -std=c++11