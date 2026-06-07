CC      = gcc
CFLAGS  = -O3 -march=native -Wall -lm
TARGET  = ising_serial
SRC     = ising_serial.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
