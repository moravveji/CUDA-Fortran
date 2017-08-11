FC = pgfortran
FCFLAGS = -c 

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = $(shell echo $(DIR_FC))/../lib
LIBS = -Mfree -Mcuda -lcublas -lblas

OBJS = dev_lib.o host_lib.o compute.o
exe  = run.exe

all: $(exe)

$(exe): $(OBJS)
	$(FC) -o $(exe) $(OBJS) -L$(DIR_LIB) $(LIBS)

dev_lib.o: dev_lib.cuf
	$(FC) $(FCFLAGS) dev_lib.cuf

host_lib.o: host_lib.cuf
	$(FC) $(FCFLAGS) host_lib.cuf

compute.o: compute.cuf
	$(FC) $(FCFLAGS) compute.cuf

clean:
	rm -rf *.o *.mod $(exe)

