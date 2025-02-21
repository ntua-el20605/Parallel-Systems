#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_fw

## Output and error files
#PBS -o run_fw.out
#PBS -e run_fw.err

## How many machines should we get?
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=01:00:00

## Start
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab13/lab2_2
#export OMP_NUM_THREADS=8
#./fw <SIZE>
./fw_sr 1024 16
# ./fw_tiled <SIZE> <BSIZE>

for threads in 1 2 4 8 16 32 64
do
for sz in 1024 2048 4096
do
for bs in 8 16 64 128 256
do
        export OMP_NUM_THREADS=$threads
        ./fw_sr $sz $bs
done
done
done
