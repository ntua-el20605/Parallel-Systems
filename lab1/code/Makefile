#!/bin/bash

## Give the Job a descriptive name
#PBS -N make_omp_gameoflive

## Output and error files
#PBS -o make_omp_gameoflive.out
#PBS -e make_omp_gameoflive.err

## How many machines should we get?
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab13/lab1
make


parlab13@scirouter:~/lab1$ cat Makefile
all: omp_gameoflive

omp_gameoflive: omp_gameoflive.c
        gcc -O3 -fopenmp -o omp_gameoflive omp_gameoflive.c

clean:
        rm omp_gameoflive