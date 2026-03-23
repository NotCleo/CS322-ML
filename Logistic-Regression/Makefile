CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lm

EXEC_GD = logistic_gd
EXEC_NEWTON = logistic_newton

SRC_GD = logistic_gd.c
SRC_NEWTON = logistic_newton.c

all: $(EXEC_GD) $(EXEC_NEWTON)

$(EXEC_GD): $(SRC_GD)
	$(CC) $(CFLAGS) -o $(EXEC_GD) $(SRC_GD) $(LDFLAGS)

$(EXEC_NEWTON): $(SRC_NEWTON)
	$(CC) $(CFLAGS) -o $(EXEC_NEWTON) $(SRC_NEWTON) $(LDFLAGS)

clean:
	rm -f $(EXEC_GD) $(EXEC_NEWTON)
