
FC = pgfortran
FCFLAGS = -c
CUDAFLAGS = -Minfo=all -Mcuda -Mfree

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = $(shell echo $(DIR_FC))/../lib

OBJS = dev_lib.o prog.o 
EXE = prog.exe

all: compile run

compile: $(EXE)

run: ./prog.exe

$(EXE): $(OBJS)
	$(FC) -o $(EXE) $(OBJS) $(LIBS)
  
%.o: %.cuf
	$(FC) $(FCFLAGS) -o $@ $<
  
clean:
	rm -f *.o *.mod $(EXE)
