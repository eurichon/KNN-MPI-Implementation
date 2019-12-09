### For more information check the [pdf][1] and the code documentation

<pre>
Execution of the code:
	* Sequential 					 		inside ./TESTER_MPI            			make
	* synchronous (tester_mpi)		 		inside ./TESTER_MPI         			make test_mpi_sync
	* ashynchronous + reduction (tester_mpi)inside ./TESTER_MPI           			make test_mpi_asyn
	* synchronous (myTester)		 		inside ./TESTER_MPI/knnring         	make mpi_test_sync
	* ashynchronous + reduction (myTester)	inside ./TESTER_MPI/knnring           	make mpi_test_asyn
	* to test blas implementation 	 		inside ./TESTER_MPI/knnring    			make blas_test 
	* to test the old  mpi version 	 		inside ./TESTER_MPI/knnring/src_old     make
	
NOTE:
For some reason while my sequential version passes the tester (./TESTER_MPI/tester) the mpi version fail in the (./TESTER_MPI/tester_mpi)
Therefore i made my own tester in (./TESTER_MPI/knnring/src/myTester) which validates that the two mpi implementation produces identical
results with the sequential. So there must be some kind of issue with the (./TESTER_MPI/tester) that was originally given.

** Each make command will rebuild from scratch cause of the .PHONY in the makefile

** To do: check and correct MPI distributed implementation
</pre>
[1]: https://github.com/eurichon/KNN-MPI-Implementation/blob/master/Exercise%20II-%208527.pdf

For any issue or difficulty running send me an email: eurichon1996@gmail.com