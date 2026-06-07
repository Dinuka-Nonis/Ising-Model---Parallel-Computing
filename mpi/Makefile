CC      = mpicc
CFLAGS  = -O3 -march=native -Wall -lm
TARGET  = ising_mpi
SRC     = ising_mpi.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	mpirun --oversubscribe -np 4 ./$(TARGET)

run1:  $(TARGET)
	mpirun --oversubscribe -np 1  ./$(TARGET)
run2:  $(TARGET)
	mpirun --oversubscribe -np 2  ./$(TARGET)
run4:  $(TARGET)
	mpirun --oversubscribe -np 4  ./$(TARGET)
run8:  $(TARGET)
	mpirun --oversubscribe -np 8  ./$(TARGET)
