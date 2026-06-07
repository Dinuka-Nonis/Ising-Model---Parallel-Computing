CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -Wall -lm
TARGET  = ising_openmp
SRC     = ising_openmp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 4

run1:  $(TARGET)
	./$(TARGET) 1
run2:  $(TARGET)
	./$(TARGET) 2
run4:  $(TARGET)
	./$(TARGET) 4
run8:  $(TARGET)
	./$(TARGET) 8
run16: $(TARGET)
	./$(TARGET) 16
