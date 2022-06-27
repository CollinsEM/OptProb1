CC = g++
NVCC = nvcc
OPT = -fopenmp -O3 #-g
INCLUDES += -I./include
CXXFLAGS += $(OPT)
LDLIBS += -lgomp
GENCODE_FLAGS = -gencode -arch=compute_61,code=sm_61
NVCC_FLAGS    = $(OPT) -m64 $(GENCODE_FLAGS) -lineinfo -Xcompiler -fopenmp #-fmad=false

SRCS = $(wildcard *.cc) #$(wildcard *.cu)
TARGETS = $(basename $(SRCS))
OBJS = $(addsuffix .o,$(TARGETS))
DEPS = $(if $(findstring clean,$(MAKECMDGOALS)),,$(addsuffix .d,$(TARGETS)))

######################################################################

default: foo_v0 foo_v1 foo_v2 foo_v3

clean:
	-rm -rf *.o *.d *~

distclean: clean
	-rm -rf $(TARGETS) *.dat

######################################################################

EXEC ?= foo_v0
NCYCLES ?= 1000
VECSIZE ?= 8192
VERBOSITY ?= #+v #+v +v
NTHREADS ?= 8

# Execute the targets using default or provided command line options
# Output is captured in v0.dat, v1.dat, or v3.dat
v0 v1 v3:
	./foo_$@ $(VERBOSITY) -ncyc $(NCYCLES) -nt $(NTHREADS) -N $(VECSIZE) >> $@.dat

# Execute the targets using default or provided command line options
# Output is captured in v2_nt#.dat.
v2:
	./foo_$@ $(VERBOSITY) -ncyc $(NCYCLES) -nt $(NTHREADS) -N $(VECSIZE) >> $@_nt$(NTHREADS).dat

run: run_v1 run_v2 run_v3

run_v0: foo_v0
	rm -f v0.dat
	VECSIZE=64 $(MAKE) v0
	VECSIZE=256 $(MAKE) v0
	VECSIZE=1024 $(MAKE) v0

run_v1: foo_v1
	rm -f v1.dat
	VECSIZE=64 $(MAKE) v1
	VECSIZE=256 $(MAKE) v1
	VECSIZE=1024 $(MAKE) v1
	VECSIZE=4096 $(MAKE) v1
	VECSIZE=16384 $(MAKE) v1
	VECSIZE=65536 $(MAKE) v1
	VECSIZE=262144 $(MAKE) v1

run_v2: run_v2_nt1 run_v2_nt2 run_v2_nt4 run_v2_nt8

run_v2_nt1: foo_v2
	rm -f v2_nt1.dat
	NTHREADS=1 VECSIZE=64 $(MAKE) v2
	NTHREADS=1 VECSIZE=256 $(MAKE) v2
	NTHREADS=1 VECSIZE=1024 $(MAKE) v2
	NTHREADS=1 VECSIZE=4096 $(MAKE) v2
	NTHREADS=1 VECSIZE=16384 $(MAKE) v2
	NTHREADS=1 VECSIZE=65536 $(MAKE) v2
	NTHREADS=1 VECSIZE=262144 $(MAKE) v2

run_v2_nt2: foo_v2
	rm -f v2_nt2.dat
	NTHREADS=2 VECSIZE=64 $(MAKE) v2
	NTHREADS=2 VECSIZE=256 $(MAKE) v2
	NTHREADS=2 VECSIZE=1024 $(MAKE) v2
	NTHREADS=2 VECSIZE=4096 $(MAKE) v2
	NTHREADS=2 VECSIZE=16384 $(MAKE) v2
	NTHREADS=2 VECSIZE=65536 $(MAKE) v2
	NTHREADS=2 VECSIZE=262144 $(MAKE) v2

run_v2_nt4: foo_v2
	rm -f v2_nt4.dat
	NTHREADS=4 VECSIZE=64 $(MAKE) v2
	NTHREADS=4 VECSIZE=256 $(MAKE) v2
	NTHREADS=4 VECSIZE=1024 $(MAKE) v2
	NTHREADS=4 VECSIZE=4096 $(MAKE) v2
	NTHREADS=4 VECSIZE=16384 $(MAKE) v2
	NTHREADS=4 VECSIZE=65536 $(MAKE) v2
	NTHREADS=4 VECSIZE=262144 $(MAKE) v2

run_v2_nt8: foo_v2
	rm -f v2_nt8.dat
	NTHREADS=8 VECSIZE=64 $(MAKE) v2
	NTHREADS=8 VECSIZE=256 $(MAKE) v2
	NTHREADS=8 VECSIZE=1024 $(MAKE) v2
	NTHREADS=8 VECSIZE=4096 $(MAKE) v2
	NTHREADS=8 VECSIZE=16384 $(MAKE) v2
	NTHREADS=8 VECSIZE=65536 $(MAKE) v2
	NTHREADS=8 VECSIZE=262144 $(MAKE) v2

run_v3: foo_v3
	./foo_v3 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE)

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

######################################################################

-include $(DEPS)

.PHONY: run_v0 v0 v1 run_v1 v2 run_v2 v3 run_v3
