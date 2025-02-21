#!/bin/bash

## Give the Job a descriptive name
#PBS -N runJacobi

## Output and error files
#PBS -o jacobi_32.out
#PBS -e jacobi_32.err

## How many machines should we get?
#PBS -l nodes=8:ppn=8

#PBS -l walltime=00:40:00

## Start
## Run make in the src folder (modify properly)

module load openmpi/1.8.3


cd /home/parallel/parlab13/lab4_2/mpi

sizes='2048 4096 6144'
threads=('1' '2' '4' '8' '16' '32' '64')
threads1=('1' '1' '2' '2' '4' '4' '8')
threads2=('1' '2' '2' '4' '4' '8' '8')

for size in $sizes
do
    for i in "${!threads[@]}"
    do
       mpirun --mca btl tcp,self -np ${threads[$i]} --map-by node ./jacobi $size $size ${threads1[$i]} ${threads2[$i]}
    done
done
