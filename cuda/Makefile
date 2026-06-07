NVCC    = nvcc
NVFLAGS = -O3 -arch=sm_89 -lm
TARGET  = ising_cuda
SRC     = ising_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 16

run8:  $(TARGET)
	./$(TARGET) 8
run16: $(TARGET)
	./$(TARGET) 16
run32: $(TARGET)
	./$(TARGET) 32
