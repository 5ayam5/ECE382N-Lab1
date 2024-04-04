test_mm: test_mm.c gen_matrix.c my_malloc.c gen_matrix.h my_malloc.h
	mpicc -g -O3 -fno-tree-vectorize -mno-avx -mno-avx2 -mno-mmx -mno-fma \
	-mno-sse2 -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 \
	test_mm.c gen_matrix.c my_malloc.c -o test_mm

run_debug:
	./test_mm 0 0 100

run_performance:
	./test_mm 1 0 100

clean:
	rm -f test_mm
