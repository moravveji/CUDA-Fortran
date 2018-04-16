FC = pgfortran
FCFLAGS = -Mcuda=cc30,cuda8.0

# Executables
access  = do_access.exe
average = do_averaging.exe

# Main functions
main_access  = access.f90
main_average = average.f90

# Kernels
kernels = $(wildcard kern_*.cuf)
objects = $(patsubst %.cuf, %.o, $(wildcard kern_*.cuf))

all: $(access)

$(access) : $(objects) access.o
	$(FC) $(FCFLAGS) -o $@ $^

%.o : %.cuf
	$(FC) $(FCFLAGS) -c $< -o $*.o

access.o: $(objects) access.f90
	$(FC) $(FCFLAGS) -c $^ -o $@

.PHONEY: clean

clean:
	rm -rf *.mod *.o $(access)
