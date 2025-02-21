#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

void kmeans(double * objects, int numCoords, int numObjs, int numClusters, double threshold, long loop_threshold, int *membership, double * clusters);

double * dataset_generation(int numObjs, int numCoords, long *rank_numObjs);

int check_repeated_clusters(int, int, double*);

double wtime(void);

extern int _debug;

#endif
parlab13@scirouter:~/lab4a$ ls *sh
make_on_queue.sh  run_on_queue.sh
parlab13@scirouter:~/lab4a$ ls *.sh
make_on_queue.sh  run_on_queue.sh
parlab13@scirouter:~/lab4a$ cat  make_on_queue.sh
#!/bin/bash

## Give the Job a descriptive name
#PBS -N make_kmeans

## Output and error files
#PBS -o make_kmeans.out
#PBS -e make_kmeans.err

## How many machines should we get?
#PBS -l nodes=1:ppn=1

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start
## Run make in the src folder (modify properly)

module load openmpi/1.8.3
cd /home/parallel/parlab13/lab4a
make
