FC = pgfortran
FCFLAGS = -c -fast
CUDAFLAGS = -Mfree -Mcuda=cuda8.0 -Minfo

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = $(shell echo $(DIR_FC))

all: compile run

compile:
	$(FC) -o run_1.exe exper_1.cuf -L$(DIR_LIB) $(CUDAFLAGS)

run:
	./run_1.exe

clean:
	rm -f *.o *.mod *.exe
