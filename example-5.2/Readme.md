## CUDA Mapped Memory

## Purpose
Practicing the allocation and access of the mapped device pointers in CUDA Fortran. This is basically called the pinned memory, which is page-locked on the host, and provides a faster access to the data, hence an improved speed up. On the host, this location has R/W access, but on the device, it only has the R access.

## Reference
[PGI CUDA Fortran Programming Guide][https://www.pgroup.com/doc/pgi17cudaforug.pdf], page 76 - 77
