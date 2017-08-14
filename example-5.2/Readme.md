## CUDA Mapped Memory

## Purpose
Practicing the allocation and access of the mapped device pointers in CUDA Fortran. This is basically called the pinned memory, which is page-locked on the host, and provides a faster access to the data, hence an improved speed up. On the host, this location has R/W access, but on the device, it only has the R access.

## Procedure to Allocate Pinned Memory
Turns out that the following steps must be taken to be able to access the page-locked memory on the host, and there is no direct way in Fortran to do so. An intervention with the `C` pointers is required:
1. First, we tell the device that we are about to use the mapped memory on the deivce, i.e. `cudaSetDeviceFlags(cudaDeviceMapHost)`. Now, the device knows where to allocate the `C`-type pointers.
2. Then, the `C`-type pointers (`type(c_ptr) :: cptr`) can be allocated on the host, using a call to `cudaHostAlloc()`.
3. Convert the C-type pointer to a Fortran-type pointer with the call to `c_f_pointer()` either for the host or device variable. Read more about this function here: https://gcc.gnu.org/onlinedocs/gfortran/C_005fF_005fPOINTER.html. 
4. use the Fortran-type device pointer for allocation and passing values. 
5. One can specify a `type(c_devptr)` C-type device pointer `a_d`, at the same memory location of the original C-type pointer `a`, and point `a_d` to `a`.
6. Then, you allocate a device Fortran-type pointer `fa_d`, and now point `fa_d` to the memory location of `a_d` on the device (using `c_f_pointer`).
7. Now, you can use the Fortran-type device pointer `fa_d` to launch a kernel, and synchronize the device
8. Then, whatever value is stored in `fa_d` can be accessed by `fa`, becaue `a`, `a_d`, `fa` and `fa_d` all point at the same location on the device memory (page-locked) and the values of the `fa` array is visible on the host for R/W.

## Reference
[PGI CUDA Fortran Programming Guide][https://www.pgroup.com/doc/pgi17cudaforug.pdf], page 76 - 77
