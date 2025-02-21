CC=mpicc
CFLAGS=-O3 -lm -Wall
RES=-DPRINT_RESULTS
CONV=-DTEST_CONV
all: jacobi jacobi_conv

jacobi: mpi_ready.c utils.c
        $(CC) $(CFLAGS) mpi_ready.c utils.c -o jacobi
jacobi_conv: mpi_ready.c utils.c
        $(CC) $(CFLAGS) $(CONV) mpi_ready.c utils.c -o jacobi_conv
clean:
        rm -f jacobi jacobi_conv
