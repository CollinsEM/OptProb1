CC = g++
NVCC = nvcc
OPT = -fopenmp -O3 #-g
INCLUDES += -I./include
CXXFLAGS += $(OPT)
LDLIBS += -lgomp
GENCODE_FLAGS = -gencode -arch=compute_61,code=sm_61
NVCC_FLAGS    = $(OPT) -m64 $(GENCODE_FLAGS) -lineinfo -Xcompiler -fopenmp #-fmad=false

SRCS = $(wildcard *.cc) foo_v3.cu #$(wildcard *.cu)
TARGETS = $(basename $(SRCS))
OBJS = $(addsuffix .o,$(TARGETS))
DEPS = $(if $(findstring clean,$(MAKECMDGOALS)),,$(addsuffix .d,$(TARGETS)))

######################################################################

default: $(TARGETS)

clean:
	-rm -rf *.o *.d *~ */*~

distclean: clean
	-rm -rf $(TARGETS) *.dat data

######################################################################

vpath %.dat data
vpath %.h include

ALLSIZES = 64 128 256 512 1024 2048 4096 8192 16384 32768
ALLTHREADS = 1 2 4 8
NCYCLES ?= 1000
VECSIZE ?= 8192
VERBOSITY ?= #+v #+v +v
NTHREADS ?= 1

data:
	mkdir -p data

# Execute the targets using default or provided command line options
# Output is captured in v0.dat, v1.dat, or v3.dat
v0 v1 v3 v4: data
	./foo_$@ $(VERBOSITY) -ncyc $(NCYCLES) -nt $(NTHREADS) -N $(VECSIZE) >> data/$@.dat

# Execute the targets using default or provided command line options
# Output is captured in v2_nt#.dat.
v2: data
	./foo_$@ $(VERBOSITY) -ncyc $(NCYCLES) -nt $(NTHREADS) -N $(VECSIZE) >> data/$@_nt$(NTHREADS).dat

run: run_v1 run_v2 run_v3 run_v4

run_v0: foo_v0
	rm -f v0.dat
	VECSIZE=64 $(MAKE) v0
	VECSIZE=256 $(MAKE) v0
	VECSIZE=1024 $(MAKE) v0

run_v1: foo_v1
	rm -f data/v1.dat
	VECSIZE=64 $(MAKE) v1
	VECSIZE=128 $(MAKE) v1
	VECSIZE=256 $(MAKE) v1
	VECSIZE=512 $(MAKE) v1
	VECSIZE=1024 $(MAKE) v1
	VECSIZE=2048 $(MAKE) v1
	VECSIZE=4096 $(MAKE) v1
	VECSIZE=8192 $(MAKE) v1
	VECSIZE=16384 $(MAKE) v1
	VECSIZE=32768 $(MAKE) v1
	VECSIZE=65536 $(MAKE) v1
	VECSIZE=131072 $(MAKE) v1
	VECSIZE=262144 $(MAKE) v1
	VECSIZE=524288 $(MAKE) v1
	VECSIZE=1048576 $(MAKE) v1

run_v2: run_v2_nt1 run_v2_nt2 run_v2_nt4 run_v2_nt8

run_v2_nt1: foo_v2
	rm -f data/v2_nt1.dat
	NTHREADS=1 VECSIZE=64 $(MAKE) v2
	NTHREADS=1 VECSIZE=128 $(MAKE) v2
	NTHREADS=1 VECSIZE=256 $(MAKE) v2
	NTHREADS=1 VECSIZE=512 $(MAKE) v2
	NTHREADS=1 VECSIZE=1024 $(MAKE) v2
	NTHREADS=1 VECSIZE=2048 $(MAKE) v2
	NTHREADS=1 VECSIZE=4096 $(MAKE) v2
	NTHREADS=1 VECSIZE=8192 $(MAKE) v2
	NTHREADS=1 VECSIZE=16384 $(MAKE) v2
	NTHREADS=1 VECSIZE=32768 $(MAKE) v2
	NTHREADS=1 VECSIZE=65536 $(MAKE) v2
	NTHREADS=1 VECSIZE=131072 $(MAKE) v2
	NTHREADS=1 VECSIZE=262144 $(MAKE) v2
	NTHREADS=1 VECSIZE=524288 $(MAKE) v2
	NTHREADS=1 VECSIZE=1048576 $(MAKE) v2

run_v2_nt2: foo_v2
	rm -f data/v2_nt2.dat
	NTHREADS=2 VECSIZE=64 $(MAKE) v2
	NTHREADS=2 VECSIZE=128 $(MAKE) v2
	NTHREADS=2 VECSIZE=256 $(MAKE) v2
	NTHREADS=2 VECSIZE=512 $(MAKE) v2
	NTHREADS=2 VECSIZE=1024 $(MAKE) v2
	NTHREADS=2 VECSIZE=2048 $(MAKE) v2
	NTHREADS=2 VECSIZE=4096 $(MAKE) v2
	NTHREADS=2 VECSIZE=8192 $(MAKE) v2
	NTHREADS=2 VECSIZE=16384 $(MAKE) v2
	NTHREADS=2 VECSIZE=32768 $(MAKE) v2
	NTHREADS=2 VECSIZE=65536 $(MAKE) v2
	NTHREADS=2 VECSIZE=131072 $(MAKE) v2
	NTHREADS=2 VECSIZE=262144 $(MAKE) v2
	NTHREADS=2 VECSIZE=524288 $(MAKE) v2
	NTHREADS=2 VECSIZE=1048576 $(MAKE) v2

run_v2_nt4: foo_v2
	rm -f data/v2_nt4.dat
	NTHREADS=4 VECSIZE=64 $(MAKE) v2
	NTHREADS=4 VECSIZE=128 $(MAKE) v2
	NTHREADS=4 VECSIZE=256 $(MAKE) v2
	NTHREADS=4 VECSIZE=512 $(MAKE) v2
	NTHREADS=4 VECSIZE=1024 $(MAKE) v2
	NTHREADS=4 VECSIZE=2048 $(MAKE) v2
	NTHREADS=4 VECSIZE=4096 $(MAKE) v2
	NTHREADS=4 VECSIZE=8192 $(MAKE) v2
	NTHREADS=4 VECSIZE=16384 $(MAKE) v2
	NTHREADS=4 VECSIZE=32768 $(MAKE) v2
	NTHREADS=4 VECSIZE=65536 $(MAKE) v2
	NTHREADS=4 VECSIZE=131072 $(MAKE) v2
	NTHREADS=4 VECSIZE=262144 $(MAKE) v2
	NTHREADS=4 VECSIZE=524288 $(MAKE) v2
	NTHREADS=4 VECSIZE=1048576 $(MAKE) v2

run_v2_nt8: foo_v2
	rm -f data/v2_nt8.dat
	NTHREADS=8 VECSIZE=64 $(MAKE) v2
	NTHREADS=8 VECSIZE=128 $(MAKE) v2
	NTHREADS=8 VECSIZE=256 $(MAKE) v2
	NTHREADS=8 VECSIZE=512 $(MAKE) v2
	NTHREADS=8 VECSIZE=1024 $(MAKE) v2
	NTHREADS=8 VECSIZE=2048 $(MAKE) v2
	NTHREADS=8 VECSIZE=4096 $(MAKE) v2
	NTHREADS=8 VECSIZE=8192 $(MAKE) v2
	NTHREADS=8 VECSIZE=16384 $(MAKE) v2
	NTHREADS=8 VECSIZE=32768 $(MAKE) v2
	NTHREADS=8 VECSIZE=65536 $(MAKE) v2
	NTHREADS=8 VECSIZE=131072 $(MAKE) v2
	NTHREADS=8 VECSIZE=262144 $(MAKE) v2
	NTHREADS=8 VECSIZE=524288 $(MAKE) v2
	NTHREADS=8 VECSIZE=1048576 $(MAKE) v2

run_v3: foo_v3
	rm -f data/v3.dat
	VECSIZE=64 $(MAKE) v3
	VECSIZE=128 $(MAKE) v3
	VECSIZE=256 $(MAKE) v3
	VECSIZE=512 $(MAKE) v3
	VECSIZE=1024 $(MAKE) v3
	VECSIZE=2048 $(MAKE) v3
	VECSIZE=4096 $(MAKE) v3
	VECSIZE=8192 $(MAKE) v3
	VECSIZE=16384 $(MAKE) v3
	VECSIZE=32768 $(MAKE) v3
	VECSIZE=65536 $(MAKE) v3
	VECSIZE=131072 $(MAKE) v3
	VECSIZE=262144 $(MAKE) v3
	VECSIZE=524288 $(MAKE) v3
	VECSIZE=1048576 $(MAKE) v3

run_v4: foo_v4
	rm -f data/v4.dat
	VECSIZE=64 $(MAKE) v4
	VECSIZE=128 $(MAKE) v4
	VECSIZE=256 $(MAKE) v4
	VECSIZE=512 $(MAKE) v4
	VECSIZE=1024 $(MAKE) v4
	VECSIZE=2048 $(MAKE) v4
	VECSIZE=4096 $(MAKE) v4
	VECSIZE=8192 $(MAKE) v4
	VECSIZE=16384 $(MAKE) v4
	VECSIZE=32768 $(MAKE) v4
	VECSIZE=65536 $(MAKE) v4
	VECSIZE=131072 $(MAKE) v4
	VECSIZE=262144 $(MAKE) v4
	VECSIZE=524288 $(MAKE) v4
	VECSIZE=1048576 $(MAKE) v4

######################################################################

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS) 

%.o: %.cc
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.d: %.cc
	$(CC) $(CXXFLAGS) $(INCLUDES) -MM $< -o $@

%.d: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -MM $< -o $@

foo_v3: foo_v3.o
	$(NVCC) $^ -o $@ $(LDLIBS)

foo_v4: foo_v4.o
	$(NVCC) $^ -o $@ $(LDLIBS)

######################################################################

-include $(DEPS)

.PHONY: run_v0 run_v1 run_v2 run_v3 run_v4
