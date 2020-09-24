# Compiler
CC = gcc
CFLAGS = -Wall -O2

# Files
SRC = $(wildcard *.c)
OBJS = $(patsubst %.c, %.o, $(SRC))

# Name of the executable
EXECUTABLE = dftw

# Compile
all: $(EXECUTABLE)

%.o: %.c
	$(CC) $(CFLAGS) -fopenmp -c $<

# Link
$(EXECUTABLE): $(OBJS)
	$(CC) -fopenmp $^ -o $@ -lm

clean:
	rm *.o
