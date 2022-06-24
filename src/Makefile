CC = g++
OPT = -O3 -fopenmp
INCLUDES += -I./include
CXXFLAGS += $(OPT)
LDLIBS += -lgomp

SRCS = $(wildcard *.cc)
OBJS = $(SRCS:%.cc=%.o)
DEPS = $(if $(findstring clean,$(MAKECMDGOALS)),,$(SRCS:%.cc=%.d))

######################################################################

default:
	$(MAKE) run

clean:
	-rm -rf *.o *.d *~

distclean: clean
	-rm -rf $(TARGETS)

######################################################################

run: run_v1

run_v0: foo_v0
	./foo_v0 -N 64
	./foo_v0 -N 512
	./foo_v0 -N 4096
	./foo_v0 -N 32768

run_v1: foo_v1
	./foo_v1 -N 64
	./foo_v1 -N 512
	./foo_v1 -N 4096
	./foo_v1 -N 32768

######################################################################

%: %.o
	$(CC) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.cc
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

%.d: %.cc
	$(CC) -MM $^ -o $@

######################################################################

-include $(DEPS)
