#! /bin/bash -l
rm *~ *.o *.mod *.exe
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 -c kernels.cuf -o kernels.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 -c main.f90 -o main.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 kernels.o main.o -o main.exe
