# Compiler
CC = gcc
CFLAGS = -Wall

# Files
SRC = $(wildcard *.c)
OBJS = $(patsubst %.c, %.o, $(SRC))

# Name of the executable
EXECUTABLE = stream

# Compile
all: $(EXECUTABLE)

%.o: %.c
	$(CC) $(CFLAGS) -fopenmp -c $<

# Link
$(EXECUTABLE): $(OBJS)
	$(CC) -fopenmp $^ -o $@ 

clean:
	rm *.o
