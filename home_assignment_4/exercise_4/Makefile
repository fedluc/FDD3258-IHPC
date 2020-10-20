# Compiler
CC = nvcc
CFLAGS = -arch=sm_30

# Files
SRC = $(wildcard *.cu)
OBJS = $(patsubst %.cu, %.o, $(SRC))

# Name of the executable
EXECUTABLE = saxpy

# Compile
all: $(EXECUTABLE)

%.o: %.cu
	$(CC) $(CFLAGS) -c $<

# Link
$(EXECUTABLE): $(OBJS)
	$(CC) $^ -o $@ 

clean:
	rm *.o
