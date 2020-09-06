

stokes: stokes.o
            nvcc -o stokes stokes.o -lm

stokes.o: stokes.cu
            nvcc -c stokes.cu

clean:
            -rm *.o a.out
