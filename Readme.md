# CUDA Fortran 90 Feature Tests

## Purpose
The purpose of this repository is to experiment with the basics of CUDA programming in Fortran 90. This repository consists of several small projects to test speed ups, memory allocation, streaming, etc. It can be used as a basic refernce for using CUDA programming in modern Fortran. It is noteworthy to mention that some of the examples are taken directly from the [PGI CUDA Fortran Programming Guide][https://www.pgroup.com/doc/pgi17cudaforug.pdf]; these folders carry the `example` in their directory name.

## Contents
+ `01-get-threadid` 
+ `example-5.1`: An extensive example for large matrix by matrix multiplication using CPU (double loop or OpenBLAS) and GPU (cuBLAS and slicing).
+ `example-5.2`: replica of the mapped-memory allocation 
+ `memory-bandwidth`: measures the effective Host2Device and Device2Host transfers for the pinned versus pageable memory.
+ `async-data-transfer`: measures the latency of four different data transfer strategies

## Requirements

## References
+ [PGI CUDA Fortran Programming Guide][https://www.pgroup.com/doc/pgi17cudaforug.pdf]
+ PGI User Forum: Accelerator Programming
