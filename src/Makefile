CC = g++
OPT = -O3 -fopenmp
INCLUDES += -I./include
CXXFLAGS += $(OPT)
LDLIBS += -lgomp

TARGETS = foo_v0
SRCS = $(addsuffix .cc,$(TARGETS))
OBJS = $(SRCS:%.cc=%.o)
DEPS = $(if $(findstring clean,$(MAKECMDGOALS)),,$(SRCS:%.cc=%.d))

default:
	echo $(TARGETS)
	$(MAKE) clean
	$(MAKE) run

clean:
	-rm -rf *.o *.d *~

distclean: clean
	-rm -rf $(TARGETS)

%: %.o
	$(CC) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.cc
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

%.d: %.cc
	$(CC) -MM $^ -o $@

-include $(DEPS)

run: run_v0

run_v0: foo_v0
	./foo_v0 -N 64
	./foo_v0 -N 512
	./foo_v0 -N 4096
	./foo_v0 -N 32768
