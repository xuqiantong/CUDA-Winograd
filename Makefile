CC=gcc
CPP=g++
AR=ar
NVCC=nvcc

CSRCS := $(shell find . -name '*.c' -not -name '._*')
COBJS := $(subst .c,.o,$(CSRCS))

CUSRCS := $(shell find . -name '*.cu' -not -name '._*')
CUOBJS := $(subst .cu,.o,$(CUSRCS))

#ALLOBJS := $(COBJS)
#ALLOBJS += $(CUOBJS)

LIBDIR := -L/usr/local/cuda/lib64 -L/home/ys646/lib/cudnn/lib64/
#-L/usr/local/cudnn5/lib64

CFLAGS= \
-I. \
-fPIC \
-I/usr/local/cuda/inlclude

CUFLAGS= \
-I. \
-Xcompiler \
-fPIC \
-I/usr/local/cuda/inlclude \
-g

LDFLAGS=-L. -lm -lpthread -lrt

#CFLAGS+=-Ofast
CFLAGS+= -O3 -ffast-math -mavx -mfma

all: Test

%.o: %.c
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

Test: $(CUOBJS) $(COBJS) 
	$(NVCC) -o Test $(CUOBJS) $(COBJS) $(LIBDIR) $(LDFLAGS) -lcudart -lcuda -lstdc++ -lcudnn -lcublas
	
clean:
	find . -name "*.o" -exec rm -f '{}' ';'
	rm -f Test


