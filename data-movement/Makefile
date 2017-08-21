FC = pgfortran
FCFLAGS = -c -fast
CUDAFLAGS = -Mfree -Mcuda=cuda8.0 -Minfo

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = $(shell echo $(DIR_FC))

all: compile run

compile:
	$(FC) -o run.exe data_move.cuf -L$(DIR_LIB) $(CUDAFLAGS)

run:
	./run.exe

clean:
	rm -f *.o *.mod *.exe
