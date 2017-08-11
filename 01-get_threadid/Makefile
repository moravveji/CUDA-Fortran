
FC = pgfortran# pgf90
FCFLAGS = -c

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC)) 
DIR_LIB = $(shell echo $(DIR_FC))/../lib
LIBS = -Mfree -Mcuda

OBJS = dev_lib.o run.o
exe  = exe

all: $(exe)

$(exe): $(OBJS)
	$(FC) -o $(exe) $(OBJS) -L$(DIR_LIB) $(LIBS)

dev_lib.o: dev_lib.cuf
	$(FC) $(FCFLAGS) dev_lib.cuf

run.o: run.cuf
	$(FC) $(FCFLAGS) run.cuf

clean: 
	rm -rf *.o *.mod $(exe)
