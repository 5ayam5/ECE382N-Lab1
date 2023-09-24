#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "gen_matrix.h"
#include "my_malloc.h"

#define MIN(a, b)         (((a) < (b)) ? (a) : (b))
#define MATIJ(i, j, cols) (((i) * (cols)) + (j))

#define MM_BLOCK_SIZE  16

// blocked matrix multiply MxP = MxN * NxP
// assumes destination matrix C is zeroed-out
// i -> M, j -> P, k -> N
void mm(double *A, double *B, double *C,
        int M, int N, int P, 
        int C_stride, int C_offset) {
  for (int i = 0; i < M; i += MM_BLOCK_SIZE) {
    for (int j = 0; j < P; j += MM_BLOCK_SIZE) {
      for (int k = 0; k < N; k += MM_BLOCK_SIZE) {
        for (int i1 = i; i1 < MIN(i + MM_BLOCK_SIZE, M); i1++) {
          for (int k1 = k; k1 < MIN(k + MM_BLOCK_SIZE, N); k1++) {
            for (int j1 = j; j1 < MIN(j + MM_BLOCK_SIZE, P); j1++) {
              C[MATIJ(i1, j1, C_stride) + C_offset] += A[MATIJ(i1, k1, N)] * B[MATIJ(k1, j1, P)];
            }
          }
        }
      }
    }
  }
}

void print_matrix(double *result, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%f ", result[MATIJ(i, j, cols)]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {

  int rank, num_procs;
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);

#ifdef DEBUG
  printf("rank %d\n", rank);
  printf("num_procs %d\n", num_procs);
#endif

  double **input_matrices;
  double *result_matrices[2];
  double *column_buf;
  double *result;
  int num_arg_matrices;

  if (argc != 4) {
    printf("usage: debug_perf test_set matrix_dimension_size\n");
    exit(1);
  }
  int debug_perf = atoi(argv[1]);
  int test_set = atoi(argv[2]);
  matrix_dimension_size = atoi(argv[3]);
  num_arg_matrices = init_gen_sub_matrix(test_set);
  int block_size = matrix_dimension_size / num_procs;

#ifdef DEBUG
  printf("num_arg_matrices %d\n", num_arg_matrices);
  printf("matrix_dimension_size %d\n", matrix_dimension_size);
  printf("block_size %d\n", block_size);
#endif
  
  // allocate arrays
  input_matrices = (double **)my_malloc(sizeof(double *) * num_arg_matrices);
  column_buf = (double *)my_malloc(sizeof(*column_buf) * matrix_dimension_size * block_size);

  for (int i = 0; i < 2; i++) {
    result_matrices[i] = (double *)my_malloc(sizeof(double) * matrix_dimension_size * block_size);
  }

  // get sub matrices
  for (int i = 0; i < num_arg_matrices; i++) {
#ifdef DEBUG
    printf("gen_sub_matrix %d on rank %d\n", i, rank);
#endif
    input_matrices[i] = (double *)my_malloc(sizeof(double) * matrix_dimension_size * block_size);
    if (gen_sub_matrix(rank, test_set, i, input_matrices[i], 0, matrix_dimension_size - 1, 1, block_size * rank, block_size * (rank + 1) - 1, 1, 1) == NULL) {
      printf("inconsistency in gen_sub_matrix\n");
      exit(1);
    }
  }

  int prev_result_idx = 0;
  for (int n = 1; n < num_arg_matrices; n++) {
    result = result_matrices[prev_result_idx];
    prev_result_idx ^= 0x1;
    double *row_matrix = (n == 1) ? input_matrices[0] : result_matrices[prev_result_idx];
    memset(result, 0, sizeof(*result) * matrix_dimension_size * block_size);

#ifdef DEBUG
    if (rank == 0) {
      printf("results[%d] <-- %s[%d] * input_matrices[%d]\n",
            prev_result_idx ^ 0x1,
            (n == 1) ? "input_matrices" : "results",
            (n == 1) ? 0 : prev_result_idx,
            n);
    }
#endif

    for (int k = 0; k < num_procs; k++) {
      for (int i = 0; i < block_size; i++) {
        memcpy(
            &column_buf[(rank * block_size + i) * block_size],
            &input_matrices[n][i * matrix_dimension_size], 
            block_size * sizeof(double));
      }
      int status = MPI_Allgather(
          MPI_IN_PLACE,
          0,
          MPI_DATATYPE_NULL,
          column_buf,
          block_size * block_size,
          MPI_DOUBLE,
          MPI_COMM_WORLD);
      if (status != MPI_SUCCESS) {
        printf("MPI Failed :( %d\n", status);
        exit(1);
      }

      if (rank == 0) {
        print_matrix(row_matrix, block_size, matrix_dimension_size);
        printf("\n");
        print_matrix(column_buf, matrix_dimension_size, block_size);
        printf("\n");
      }

      mm(
          row_matrix, column_buf, result, 
          block_size, matrix_dimension_size, block_size,
          matrix_dimension_size, k * block_size
      );
    }
    if (rank == 0) {
      print_matrix(result, block_size, matrix_dimension_size);
      printf("\n");
    }
  }

  if (debug_perf == 0) {
    // print each of the sub matrices
    for (int i = 0; i < num_arg_matrices; i++) {
      if (rank == 0) {
        printf("argument matrix %d\n", i);
        fflush(stdout);
      }
      for (int j = 0; j < num_procs; j++) {
        if (j == rank) {
          print_matrix(input_matrices[i], block_size, matrix_dimension_size);
          fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      if (rank == 0) {
        printf("\n");
        fflush(stdout);
      }
    }

    if (rank == 0) {
      printf("result matrix\n");
      fflush(stdout);
    }
    for (int j = 0; j < num_procs; j++) {
      if (j == rank) {
        print_matrix(result, block_size, matrix_dimension_size);
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if (rank == 0) {
      printf("\n");
      fflush(stdout);
    }
  } else { // benchmark mode -- sum all
    double block_sum = 0.0;
    for (int i = 0; i < matrix_dimension_size * block_size; i++) {
      block_sum += result[i];
    }

    double all_sum;
    MPI_Reduce(&block_sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      printf("%f\n", all_sum);
    }
  }

  if (rank == num_procs - 1) {
    printf("time: %f\n", MPI_Wtime() - start_time);
  }
  MPI_Finalize();
  return 0;
}

