Execution of the code:
	1)Sequential 					inside ./TESTER_MPI            			make
	2)synchronous 					inside 	./TESTER_MPI           			make test_mpi
	3-4)ashynchronous + reduction 	inside ./TESTER_MPI            			make test_asyc_mpi
	5)to test blas implementation 	inside ./TESTER_MPI/knnring    			make blas_test 
	6)to test the old  mpi version 	inside ./TESTER_MPI/knnring/src_old     make
	
NOTE:
if you type make inside ./TESTER_MPI/knnring you can verify that the old version was working correctly regarding the values
as it prints the same result with the sequential

** Each make command will rebuild from scratch cause of the .PHONY in the makefile

** To do: check and correct MPI distributed implementation

