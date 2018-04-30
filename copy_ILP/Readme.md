# Copying Data Uisng Instruction-Level Parallelism (ILP)

## Purpose
This simple example shows how to copy an array (size is power of 2) between global memory using a launch of a single block of `bs` threads, and employ a `for`-loop in the kernel to exploit ILP. Then, the bandwidth (BW) is measured for this and a similar operation using the shared memory.

## Why?
Assume at a given phase of your application you need to copy many arrays simulataneously. One possibility is the following: You can stream your copying operations concurrently through many CUDA streams but assign each copy operation to the launch of a single kernel and all available threads within that single block. Explicitly, instead of 
```fortran
call copy_shared_mem<<<Num_Blocks, Threads_per_Block>>> (source_d, target_d, 0)
```
you can launch your kernel accordingly:
```fortran
call copy_shared_mem<<<1, Threads_per_Block>>> (source_d, target_d, Num_Blocks)
```
The reason is that the calling kernel has an explicit loop which jumps over the whole source/target arrays and copies elements whose index are a multiple of the `Num_Blocks`. E.g.
```fortran
    do k = 0, nb - 1
       temp(k + 1) = src(i + k * bs)
    enddo

```

## Result
The bandwidth of the copy operation from global to global memory is 1.59 GB/sec, but the same copy using an intermediate array declared in the share memory offers a bandwidth of 5.91 GB/sec on a Kepler K40c device.
