CC= mpicc
CFLAGS= -g
LIBS = -lm -lopenblas -lpthread -lgfortran
MAIN= knn

All: knn.o main.o
	$(CC) $(CFLAGS) -o $(MAIN) knn.o main.o -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib $(LIBS)  
	mpirun -np 6 ./$(MAIN) 21 10 7
	
knn.o: knn.c knn.h
	$(CC) $(CFLAGS) -c knn.c 
	
main.o: main.c knn.h
	$(CC) $(CFLAGS) -c main.c

clean: 
	$(RM) count *.o *~ $(MAIN)
	