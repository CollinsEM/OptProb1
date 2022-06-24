CC = g++
OPT = -O3 -fopenmp
INCLUDES += -I./include
CXXFLAGS += $(OPT)
LDLIBS += -lgomp

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

run: run_v1 run_v2

run_v0: foo_v0
	./foo_v0 -N 64
	./foo_v0 -N 512
	./foo_v0 -N 4096
	./foo_v0 -N 32768

run_v1: foo_v1
	./foo_v1 -v -ncyc $(NCYCLES) -N 64
	./foo_v1 -v -ncyc $(NCYCLES) -N 512
	./foo_v1 -v -ncyc $(NCYCLES) -N 4096
	./foo_v1 -v -ncyc $(NCYCLES) 1000 -N 32768

run_v2: foo_v2
#	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 1
	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 1 -pavg
#	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 2
	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 2 -pavg
#	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 4
	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 4 -pavg
#	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 8
	./foo_v2 -ncyc $(NCYCLES) -N 32768 -nt 8 -pavg

run_v3: foo_v3
	./foo_v3 -N 64
	./foo_v3 -N 512
	./foo_v3 -N 4096
	./foo_v3 -N 32768

######################################################################

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS) 

%.o: %.cc
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

%.d: %.cc
	$(CC) -MM $^ -o $@

######################################################################

-include $(DEPS)
