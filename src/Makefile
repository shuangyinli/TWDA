export CC = gcc
export CXX = g++
export CFLAGS = -Wall -g -O3 -pthread

INSTALL_PATH=bin/
BIN = twda
OBJ = twda-estimate.o  twda-inference.o twda-learn.o
.PHONY: clean all

all: $(BIN)

twda:twda.c twda-estimate.o twda-inference.o twda-learn.o twda.h
twda-estimate.o:twda-estimate.c utils.h twda.h twda-estimate.h
twda-inference.o: twda-inference.c utils.h twda-inference.h twda.h
twda-learn.o: twda-learn.c utils.h twda-learn.h twda.h

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(filter %.cpp %.c, $^)

install:
	cp -f -r $(BIN) $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
