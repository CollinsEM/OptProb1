CC = g++
NVCC = nvcc
OPT = -O3 -fopenmp
INCLUDES += -I./include
CXXFLAGS += $(OPT)
LDLIBS += -lgomp
GENCODE_FLAGS = -gencode -arch=compute_61,code=sm_61
NVCC_FLAGS    = $(OPT) -m64 $(GENCODE_FLAGS) -lineinfo -Xcompiler -fopenmp #-fmad=false

SRCS = $(wildcard *.cc)
OBJS = $(SRCS:%.cc=%.o)
DEPS = $(if $(findstring clean,$(MAKECMDGOALS)),,$(SRCS:%.cc=%.d))

TARGETS = $(SRCS:%.cc=%)
######################################################################

default:
	$(MAKE) run

clean:
	-rm -rf *.o *.d *~

distclean: clean
	-rm -rf $(TARGETS)

######################################################################

NCYCLES = 1000
VECSIZE = 8192
VERBOSITY = #+v #+v +v
NTHREADS = 8

run: run_v1 run_v2 run_v3

run_v0: foo_v0
	./foo_v0 -N 64
	./foo_v0 -N 512
	./foo_v0 -N 4096
	./foo_v0 -N 32768

run_v1: foo_v1
	./foo_v1 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE)

run_v2: foo_v2
#	./foo_v2 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE) -nt $(NTHREADS)
	./foo_v2 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE) -nt 1 -pavg
	./foo_v2 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE) -nt 2 -pavg
	./foo_v2 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE) -nt 4 -pavg
	./foo_v2 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE) -nt 8 -pavg

run_v3: foo_v3
	./foo_v3 $(VERBOSITY) -ncyc $(NCYCLES) -N $(VECSIZE)

######################################################################

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS) 

%.o: %.cc
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@

%.d: %.cc
	$(CC) -MM $^ -o $@

foo_v3: foo_v3.o
	$(NVCC) $^ -o $@ $(LDLIBS)

######################################################################

-include $(DEPS)
