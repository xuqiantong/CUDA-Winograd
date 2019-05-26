CC=gcc
CPP=g++
AR=ar
NVCC=nvcc

CSRCS := $(shell find . -name '*.c' -not -name '._*')
COBJS := $(subst .c,.o,$(CSRCS))

CUSRCS := $(shell find . -name '*.cu' -not -name '._*')
CUOBJS := $(subst .cu,.o,$(CUSRCS))

LIBDIR := -L/usr/local/cuda/lib64

CUFLAGS= \
-I. \
-Xcompiler \
-fPIC

LDFLAGS=-L. -lm -lpthread -lrt

all: Test

%.o: %.c
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

Test: $(CUOBJS) $(COBJS)
	$(NVCC) -o Test $(CUOBJS) $(COBJS) $(LIBDIR) $(LDFLAGS) -lcudart -lcuda -lcublas -lcudnn

clean:
	find . -name "*.o" -exec rm -f '{}' ';'
	rm -f Test
