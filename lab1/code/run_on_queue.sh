#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_omp_gameoflive

## Output and error files
#PBS -o run_omp_gameoflive.out
#PBS -e run_omp_gameoflive.err

## How many machines should we get?
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:15:00

## Start
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab13/lab1

export OMP_NUM_THREADS=8
./omp_gameoflive 4096 1000