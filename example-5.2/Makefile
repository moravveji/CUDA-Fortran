
FC = pgfortran
FCFLAGS = -c -g
CUDAFLAGS = -Minfo=all -Mcuda -Mfree

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = $(shell echo $(DIR_FC))/../lib

OBJS = dev.o page77.o 
EXE = run.exe

all: compile run

compile: $(EXE)

run: ./run.exe

$(EXE): $(OBJS)
	$(FC) -o $(EXE) $(OBJS) $(CUDAFLAGS)
  
%.o: %.cuf
	$(FC) $(FCFLAGS) -o $@ $<
  
clean:
	rm -f *.o *.mod $(EXE)
