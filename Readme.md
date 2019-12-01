### For more information check the [pdf][1] and the code documentation

<pre>
Execution of the code:
	* Sequential 					inside ./TESTER_MPI            			make
	* synchronous 					inside 	./TESTER_MPI           			make test_mpi
	* ashynchronous + reduction 	                inside ./TESTER_MPI            			make test_asyc_mpi
	* to test blas implementation 	                inside ./TESTER_MPI/knnring    			make blas_test 
	* to test the old  mpi version 	                inside ./TESTER_MPI/knnring/src_old             make
	
NOTE:
if you type make inside ./TESTER_MPI/knnring you can verify that the old version was working correctly regarding the values
as it prints the same result with the sequential

** Each make command will rebuild from scratch cause of the .PHONY in the makefile

** To do: check and correct MPI distributed implementation
</pre>
[1]: https://github.com/eurichon/KNN-MPI-Implementation/blob/master/Exercise%20II-%208527.pdf
