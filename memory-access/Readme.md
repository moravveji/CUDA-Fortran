# Memory Access

## Purpose
This basic example tries to measure the bandwidth (BW) of accessding *misaligned* and *strided* data through the *global* and *texture memory*. For details, one may refer to Section 3.2 in the book: [CUDA Fortran](https://www.elsevier.com/books/cuda-fortran-for-scientists-and-engineers/ruetsch/978-0-12-416970-8).

## Dependencies and Compilation
This example was compiled with `PGI v.17.4` compiler and `CUDA 8.0` toolkit that ships with the compiler. To compile, just execute `make` to invoke the `Makefile`.
