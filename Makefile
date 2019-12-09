CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 4
MAIN = test_sequential



.PHONY: test_sequential test_mpi_sync test_mpi_asyn
test_sequential:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(CC) tester.c knnring_sequential.a -o $@ -lopenblas -lpthread -lgfortran -lm
	./test_sequential

test_mpi_sync:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring_syn_asyn.h ../; cd ..
	$(MPICC) tester_mpi.c knnring_mpi.a -o $@ -lopenblas -lpthread -lgfortran -lm
	$(MPIRUN) ./test_mpi_sync

test_mpi_asyn:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring_syn_asyn.h ../; cd ..
	$(MPICC) tester_mpi.c knnring_mpi_asyc.a -o $@ -lopenblas -lpthread -lgfortran -lm
	$(MPIRUN) ./test_mpi_asyn

clean: 
	$(RM) count *.a *.o *~ test_sequential test_mpi_asyn test_mpi_sync