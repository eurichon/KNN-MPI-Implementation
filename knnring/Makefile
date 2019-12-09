CC=gcc
MPICC=mpicc
CFLAGS=-g -Wall
LIBS =-lm -lopenblas -lpthread -lgfortran
MAIN= knnring_sequential
DEPS= knnring.h

.PHONY: lib blas_test mpi_test_asyn mpi_test_sync
All: src/knnring_sequential.o src/main.o
	$(CC) $(CFLAGS) -o $(MAIN) src/knnring_sequential.o src/main.o -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib $(LIBS)
	./$(MAIN)
	
knn.o: src/knnring_sequential.c inc/knnring.h
	$(CC) $(CFLAGS) -c src/knnring_sequential.c 
	
main.o: src/main.c inc/knnring.h
	$(CC) $(CFLAGS) -c src/main.c

blas_test: src/blas_test.c
	$(CC) src/blas_test.c -o blas_test -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib $(LIBS)
	./blas_test 1000 1000 100

mpi_test_sync: src/knnring_mpi.c src/myTester.c
	$(MPICC) -o mpi_test_sync src/knnring_mpi.c src/myTester.c -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib $(LIBS)
	mpirun -np 4 ./mpi_test_sync 500 20 50
	
mpi_test_asyn: src/knnring_mpi_asyc.c src/myTester.c
	$(MPICC) -o mpi_test_asyn src/knnring_mpi_asyc.c src/myTester.c -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib $(LIBS)
	mpirun -np 4 ./mpi_test_asyn 500 20 50

lib: src/knnring_sequential.o inc/knnring.h
	$(CC) -c -o src/knnring_sequential.o src/knnring_sequential.c 
	ar rcs lib/knnring_sequential.a src/knnring_sequential.o
	rm src/knnring_sequential.o	
	@echo Serial Static Library created in knnring/Lib!
	
	$(MPICC) -c -o src/knnring_mpi.o src/knnring_mpi.c 
	ar rcs lib/knnring_mpi.a src/knnring_mpi.o
	rm src/knnring_mpi.o	
	@echo Mpi Static Library created in knnring/Lib!
	
	$(MPICC) -c -o src/knnring_mpi_asyc.o src/knnring_mpi_asyc.c 
	ar rcs lib/knnring_mpi_asyc.a src/knnring_mpi_asyc.o
	rm src/knnring_mpi_asyc.o	
	@echo Mpi Static Library created in knnring/Lib!
	
clean:
	$(RM) count lib/*.a lib/*~ src/*.o src/*~ $(MAIN)