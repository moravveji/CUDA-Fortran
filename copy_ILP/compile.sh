#!/bin/bash -l
rm -f *~ *.o *.mod *.exe
pgfortran -g -O2 -Mcuda=cc35,cuda8.0,ptxinfo,keepptx -c kerns.cuf -o kerns.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0,ptxinfo,keepptx -c main.f90 -o main.o
pgfortran -g -O2 -Mcuda=cc35,cuda8.0,ptxinfo,keepptx kerns.o main.o -o main.exe
