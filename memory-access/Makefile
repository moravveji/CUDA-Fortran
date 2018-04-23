FC = pgfortran
FCFLAGS = -g -O2 -Mcuda=fastmath,cc35,cuda8.0 #-Mlarge_array

# Executables
access  = do_access.exe
average = do_averaging.exe

# Main functions
main_access  = access.f90
main_average = average.f90

# Kernels
kernels = $(wildcard kern_*.cuf)
objects = $(patsubst %.cuf, %.o, $(wildcard kern_*.cuf))
 
all: $(access) $(average)

$(access) : $(objects) access.o
	$(FC) $(FCFLAGS) -o $@ $^

$(average) : kern_averages.o average.o
	$(FC) $(FCFLAGS) -o $@ $^

%.o : %.cuf
	$(FC) $(FCFLAGS) -c $< -o $*.o

access.o: access.f90
	$(FC) $(FCFLAGS) -c $^ -o $@

average.o: average.f90
	$(FC) $(FCFLAGS) -c $^ -o $@

.PHONEY: clean

clean:
	rm -rf *~ *.mod *.o $(access) $(average) 
