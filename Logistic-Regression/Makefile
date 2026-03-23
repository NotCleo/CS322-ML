# Compiler settings
CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm

# Executable names
EXEC_GD = logistic_gd
EXEC_NEWTON = logistic_newton

# Source files
SRC_GD = logistic_gd.c
SRC_NEWTON = logistic_newton.c

# Default target compiles both
all: $(EXEC_GD) $(EXEC_NEWTON)

# Rule for Gradient Descent
$(EXEC_GD): $(SRC_GD)
	$(CC) $(CFLAGS) -o $(EXEC_GD) $(SRC_GD) $(LDFLAGS)

# Rule for Newton's Method
$(EXEC_NEWTON): $(SRC_NEWTON)
	$(CC) $(CFLAGS) -o $(EXEC_NEWTON) $(SRC_NEWTON) $(LDFLAGS)

# Clean up binaries
clean:
	rm -f $(EXEC_GD) $(EXEC_NEWTON)
