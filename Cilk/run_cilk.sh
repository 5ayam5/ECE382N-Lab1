#!/bin/bash
# filename: run_mpi.sh
#SBATCH -J mpi_mm # job name
#SBATCH -o ../logs/mpi_mm.o%j # output and error file name (%j expands to jobID)
#SBATCH -e ../logs/mpi_mm.e%j # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of mpi tasks requested
#SBATCH -N 1 # number of mpi nodes requested
#SBATCH -p development # queue (partition) -- normal, development, etc.
#SBATCH -t 00:10:00 # run time (hh:mm:ss) - 30 seconds
#SBATCH --mail-user=ss225962@utexas.edu
#SBATCH --mail-type=all # send email at begin and end of job
CILK_NWORKERS=1 ./test_mm 1 0 16