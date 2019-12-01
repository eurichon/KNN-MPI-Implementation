CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 4
MAIN = test_sequential



.PHONY: test_sequential test_mpi test_asyc_mpi
test_sequential:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(CC) tester.c knnring_sequential.a -o $@ -lopenblas -lpthread -lgfortran -lm
	./test_sequential



test_mpi:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring_syn_asyn.h ../; cd ..
	$(MPICC) tester_mpi.c knnring_mpi.a -o $@ -lopenblas -lpthread -lgfortran -lm
	$(MPIRUN) ./test_mpi



test_asyc_mpi:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring_syn_asyn.h ../; cd ..
	$(MPICC) tester_mpi.c knnring_mpi_asyc.a -o $@ -lopenblas -lpthread -lgfortran -lm
	$(MPIRUN) ./test_asyc_mpi



clean: 
	$(RM) count *.a *.o *~ $(MAIN)