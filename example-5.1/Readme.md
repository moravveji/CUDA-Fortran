
# CPU and GPU Matrix by Matrix Multiplication

## Purpose
This simple test case carries out the basic multiplication of two matrices `A` and `B`, and stores the result in the third matrix `C`. For this test, the following four functions are called:

+ `do_mul_gpu`: This is basically the example algorithm from the [PGI CUDA Fortran Programming Guide and Reference][https://www.pgroup.com/doc/pgi17cudaforug.pdf]. To compute each block of the `C` matrix, the input `A` and `B` matrices are sliced into 16x16 sub-matrices, and then multiplied. As the manual claims, this is algorithm is still far from efficient. The computation is indeed offloaded to the device.
+ `do_blas`: which basically calls the BLAS function `sgemm` for a single precision matrix by matrix multiplication, setting `alpha=1.0`, and `beta=0.0`. This is done on a single core CPU.
+ `do_cublas`: which basically calls the cuBLAS equivalent of sgemm. The only difference is that inside this functio, the cuBLAS library is used as `use cublas`. Indeed, the computations are offloaded to the GPU device.
+ `do_cpu`: which is the *dumbest-ever* way of doing the matrix multiplication, i.e. through a nested double `for` loops, and a dot-product of the i-th row of the matrix `A` with the j-th column of the matrix `B` to get `C(i,j)`. However, it is always instructive to have this routine in, and see with bare eyes how it struggles!

The input matrices `A` and `B` are randomly initiated per each test, but the same matrices are used for all routines. The full multiplication in each of the above functions are repeated `N` times, and the resulting runtime is averaged out, and printed on the screen. All functions are timed, using the `cpu_time()` intrinsic function call; most importantly, the timing is done only around the for iteration for-loop, and exlcudes the data copying between the host and the device. 

## Comparison Tests
The following benchmarking results are averages for `N=10` iterations, with matrix sizes of `A(3200, 4800)`, `B(4800, 6400)` and of course `C(3200, 6400)`. I used an NVIDIA GPU K20Xm attached to a node with single core type of E52630 Westmere compute node. In all test, the host-device data transfer are *not* timed. Then, I have used the time spent by the call to the OpenBLAS `sgemm` as the baseline for the speed up comparison. 

Function | Which? | Time (sec) | Speed up
--- | --- | --- | ---
`do_blas` | CPU | 5.3669 | 1.0x
`do_cpu` | CPU | 422 | awful
`do_mul_gpu` | GPU | 0.0331 | 162x
`do_cublas` | GPU | 0.8209 | 6.5x

## Requirements
+ PGI 17.4 compiler
+ OpenBLAS math library (0.2.13-GCC-4.9.2-LAPACK-3.5.0)
+ cuBLAS library

