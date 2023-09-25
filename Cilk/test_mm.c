#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cilk/cilk.h>
#include "gen_matrix.h"
#include "my_malloc.h"

#ifdef SILENCING_INTELLISENSE
#define _Cilk_spawn
#define _Cilk_sync
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MATIJ(i, j, cols) (((i) * (cols)) + (j))

#define MM_BLOCK_SIZE 64

// blocked matrix multiply MxP = MxN * NxP
// assumes destination matrix C is zeroed-out
// i -> M, j -> P, k -> N
void mm(double *A, double *B, double *C,
        int M, int N, int P,
        int A_stride, int A_offset,
        int B_stride, int B_offset,
        int C_stride, int C_offset)
{
  for (int i = 0; i < M; i += MM_BLOCK_SIZE) {
    for (int j = 0; j < P; j += MM_BLOCK_SIZE) {
      for (int k = 0; k < N; k += MM_BLOCK_SIZE) {
        for (int i1 = i; i1 < MIN(i + MM_BLOCK_SIZE, M); i1++) {
          for (int k1 = k; k1 < MIN(k + MM_BLOCK_SIZE, N); k1++) {
            for (int j1 = j; j1 < MIN(j + MM_BLOCK_SIZE, P); j1++) {
              C[MATIJ(i1, j1, C_stride) + C_offset] += 
                  A[MATIJ(i1, k1, A_stride) + A_offset] * 
                  B[MATIJ(k1, j1, B_stride) + B_offset];
            }
          }
        }
      }
    }
  }
}

void print_matrix(double *result, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
      printf("%f ", result[MATIJ(i, j, cols)]);
    printf("\n");
  }
}

double block_sum(double *result, int rows, int cols)
{
  double sum = 0.0;
  for (int i = 0; i < rows * cols; i++)
    sum += result[i];
  return sum;
}

int main(int argc, char *argv[])
{
  time_t start_time;
  time(&start_time);

  int num_procs = __cilkrts_get_nworkers();
#if DEBUG
  printf("num procs %d\n", num_procs);
#endif
  fflush(stdout);

  double **input_matrices;
  double *result_matrices[2];
  double *result;
  int num_arg_matrices;

  if (argc != 4)
  {
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
  fflush(stdout);

  // allocate arrays
  input_matrices = (double **)my_malloc(sizeof(double *) * num_arg_matrices);

  for (int i = 0; i < 2; i++) {
    result_matrices[i] = (double *)my_malloc(sizeof(double) * matrix_dimension_size * matrix_dimension_size);
  }

  // get sub matrices
  for (int i = 0; i < num_arg_matrices; ++i) {
    input_matrices[i] = (double *)my_malloc(sizeof(double) * matrix_dimension_size * matrix_dimension_size);
    if (gen_sub_matrix(0, test_set, i, input_matrices[i], 0, matrix_dimension_size - 1, 1, 0, matrix_dimension_size - 1, 1, 1) == NULL) {
      printf("inconsistency in gen_sub_matrix\n");
      exit(1);
    }
  } 

  int prev_result_idx = 0;
  for (int n = 1; n < num_arg_matrices; n++) {
    result = result_matrices[prev_result_idx];
    prev_result_idx ^= 0x1;
    double *row_matrix = (n == 1) ? input_matrices[0] : result_matrices[prev_result_idx];
    memset(result, 0, sizeof(*result) * matrix_dimension_size * matrix_dimension_size);

    for (int i = 0; i < num_procs; i++) {
      for (int j = 0; j < num_procs; j++) {
        cilk_spawn mm(
            row_matrix, input_matrices[n], result, 
            block_size, matrix_dimension_size, block_size,
            matrix_dimension_size, i * block_size * matrix_dimension_size,
            matrix_dimension_size, j * block_size,
            matrix_dimension_size, (i * matrix_dimension_size + j) * block_size
        );
      }
    }
    cilk_sync;
  }

  if (debug_perf == 0)
  {
    // print each of the sub matrices
    for (int i = 0; i < num_arg_matrices; i++)
    {
      printf("argument matrix %d\n", i);
      print_matrix(input_matrices[i], matrix_dimension_size, matrix_dimension_size);
      printf("\n");
    }

    printf("result matrix\n");
    print_matrix(result, matrix_dimension_size, matrix_dimension_size);
    fflush(stdout);
  }
  else
  { // benchmark mode -- sum all
    double *block_sums = my_malloc(sizeof(double) * num_procs);
    for (int rank = 0; rank < num_procs; rank++)
      block_sums[rank] = cilk_spawn block_sum(&result[rank * block_size * matrix_dimension_size], block_size, matrix_dimension_size);
    cilk_sync;

    double all_sum = 0;
    for (int rank = 0; rank < num_procs; rank++)
      all_sum += block_sums[rank];
    
    printf("%f\n", all_sum);
  }

  return 0;
}
