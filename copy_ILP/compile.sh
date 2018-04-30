#!/bin/bash -l
rm -f *~ *.o *.mod *.exe
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 -c kerns.cuf -o kerns.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 -c main.f90 -o main.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0 kerns.o main.o -o main.exe
