#! /bin/bash -l
rm *~ *.o *.mod *.exe
set -x
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 -c kernels.cuf -o kernels.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 -c main.f90 -o main.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 kernels.o main.o -o main.exe

pgfortran -g -O2 -Mcuda=cc35,cuda8.0 -c tex_vs_intent.f90 -o tex_vs_intent.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 kernels.o tex_vs_intent.o -o test_intent.exe
