# all       - make opt, dbg, and test
# opt       - build optimized shared library using REAL_T as the real type
# dbg       - build debugging shared library using REAL_T as the real type
# test      - build and run unit tests
# install   - install headers and shared libraries
# headers   - install headers only; you must define NN_HEADER_ONLY when compiling
# clean     - clean folders used by opt, dbg, and test
# uninstall - remove installed headers and shared libraries

# BEGIN VARIABLES

# Name of optimized lib
OPT := nnlib

# Name of debugging lib
DBG := $(OPT)_dbg

# Name of test executable
TST := $(OPT)_test

# Which linear algebra acceleration library to use on CPU
ACCEL_CPU := auto
# ACCEL_CPU := openblas
# ACCEL_CPU := none

# Whether to use a linear algebra library on GPU
ACCEL_GPU := auto
# ACCEL_GPU := none

# Location of the linear algebra library to use on CPU (required for GPU)
CPU_BLAS := /usr/local/lib/libopenblas.so

# Which real type to use when precompiling shared libraries
REAL_T := double

# Prefix of where to install headers and shared libraries
PREFIX := /usr/local

# Compiler flags
CXXFLAGS := -Wall

# END VARIABLES

ifneq ($(REAL_T),none)
    override CXXFLAGS += -DNN_REAL_T=$(REAL_T)
endif

override CXXFILES := $(wildcard src/*.cpp) $(wildcard src/**/*.cpp)
override OPTFILES := $(CXXFILES:src/%.cpp=obj/%.o)
override DBGFILES := $(CXXFILES:src/%.cpp=obj/dbg/%.o)
override DEPFILES := $(OPTFILES:%.o=%.d) $(DBGFILES:%.o=%.d)
override CXXFLAGS += -std=c++11 -Iinclude

ifeq ($(ACCEL_CPU)$(shell uname -s),autoDarwin)
    override CXXFLAGS += -DNN_ACCEL_CPU
    override LDFLAGS += -framework Accelerate
else
ifneq ($(ACCEL_CPU),none)
    override CXXFLAGS += -DNN_ACCEL_CPU
    override LDFLAGS += -lopenblas
endif
endif

ifneq ($(ACCEL_GPU),none)
ifneq ($(shell which nvcc),)
    override CXXFLAGS += -DNN_ACCEL_GPU
    override LDFLAGS += -lnvblas
endif
endif

ifeq ($(shell uname -s),Darwin)
    override LDFLAGS += -dynamiclib
    override OPTLIB := lib$(OPT).dylib
    override DBGLIB := lib$(DBG).dylib
else
    override LDFLAGS += -shared
    override OPTLIB := lib$(OPT).so
    override DBGLIB := lib$(DBG).so
endif

override OPTFLAGS := $(CXXFLAGS) -DNN_OPT -O3
override DBGFLAGS := $(CXXFLAGS) -g

override TSTFILES := $(wildcard test/*.cpp) $(wildcard test/**/*.cpp)
override TSTFILES := $(TSTFILES:test/%.cpp=obj/test/%.o)

override HXXFILES := $(shell find include -type f)
override HXXFILES := $(HXXFILES:%=$(PREFIX)/%)

all: opt dbg test

opt: lib/$(OPTLIB)
lib/$(OPTLIB): $(OPTFILES)
	@mkdir -p $(dir $@)
	$(CXX) -fPIC $(OPTFILES) $(LDFLAGS) -o $@
obj/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -fPIC $< $(OPTFLAGS) -MMD -c -o $@

dbg: lib/$(DBGLIB)
lib/$(DBGLIB): $(DBGFILES)
	@mkdir -p $(dir $@)
	$(CXX) -fPIC $(DBGFILES) $(LDFLAGS) -o $@
obj/dbg/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -fPIC $< $(DBGFLAGS) -MMD -c -o $@

test: dbg bin/$(TST) bin/nvblas.conf
	NVBLAS_CONFIG_FILE=bin/nvblas.conf ./bin/$(TST)
bin/$(TST): $(TSTFILES)
	@mkdir -p $(dir $@)
	$(CXX) $(TSTFILES) -Wl,-rpath,lib -Llib -l$(DBG) -o $@
obj/test/%.o: test/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(DBGFLAGS) -MMD -c -o $@
bin/nvblas.conf:
	@echo "NVBLAS_CPU_BLAS_LIB" $(CPU_BLAS) > $@
	@echo "NVBLAS_GPU_LIST ALL" >> $@
	@echo "NVBLAS_TILE_DIM 2048" >> $@
	@echo "NVBLAS_AUTOPIN_MEM_ENABLED" >> $@

install: opt dbg headers
	cp lib/$(OPTLIB) $(PREFIX)/lib/
	cp lib/$(DBGLIB) $(PREFIX)/lib/

headers: $(HXXFILES)
$(PREFIX)/%: %
	@mkdir -p $(dir $@)
	cp $< $@

clean:
	rm -rf bin lib obj

uninstall:
	rm -rf $(PREFIX)/include/nnlib*
	rm -f $(PREFIX)/lib/$(OPTLIB) $(PREFIX)/lib/$(DBGLIB)

.PHONY: all opt dbg test install headers clean uninstall

-include $(DEPFILES)
