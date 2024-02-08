# Compiler will default to standard C++ compiler on the system (e.g. g++ or clang++)
CC      = mpic++
CFLAGS  = -std=c++20 -Wall -pthread -Xpreprocessor -fopenmp
LDFLAGS = -lomp


all: mpi_setup_test


mpi_setup_test: mpi_setup_test.o
	$(CC) $(CFLAGS) -o mpi_setup_test mpi_setup_test.o $(LDFLAGS)


mpi_setup_test.o: mpi_setup_test.cpp
	$(CC) $(CFLAGS) -c mpi_setup_test.cpp


clean:
	rm -f mpi_setup_test mpi_setup_test.o