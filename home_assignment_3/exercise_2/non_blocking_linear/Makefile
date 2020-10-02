# Compiler
CC = cc
CFLAGS = -Wall

# Files
SRC = $(wildcard *.c)
OBJS = $(patsubst %.c, %.o, $(SRC))

# Name of the executable
EXECUTABLE = pi

# Compile
all: $(EXECUTABLE)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

# Link
$(EXECUTABLE): $(OBJS)
	$(CC) $^ -o $@ 

clean:
	rm *.o
