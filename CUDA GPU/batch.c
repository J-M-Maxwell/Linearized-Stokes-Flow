#!/bin/bash
#PBS -N stokes.c
#PBS -m abe
#PBS -M jmaxwell2020@u.northwestern.edu
#PBS -l walltime=00:15:00
#PBS -q batch
cd $PBS_O_WORKDIR
./stokes 128 1.0 1 0.4 1e-9 100000
