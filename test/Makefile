BIN:=../bin
OBJ:=../obj
INC:=../include
OUT:=nnlib_test
CFLAGS:=-Wall -std=c++11 -I$(INC) -DHAS_BLAS
LFLAGS:=
CXX:=g++

GNU_VERSION:=$(shell $(CXX) --version 2>/dev/null | grep ^g++ | sed 's/^.* //g')
ifneq ($(GNU_VERSION),)
	CFLAGS+=-flax-vector-conversions
endif

UNAME:=$(shell uname -s)
ifeq ($(UNAME),Darwin)
	LFLAGS+=-framework Accelerate
else
	LFLAGS+=-L/usr/local/lib -lopenblas -lpthread
endif

CFLAGS_OPT:=$(CFLAGS) -O3 -DOPTIMIZE
CFLAGS_DBG:=$(CFLAGS) -g -DDEBUG

LFLAGS_OPT:=$(LFLAGS)
LFLAGS_DBG:=$(LFLAGS)

CPP_FILES:=$(wildcard *.cpp) $(wildcard **/*.cpp)
DEP_FILES:=$(CPP_FILES:%.cpp=$(OBJ)/opt/%.d) $(CPP_FILES:%.cpp=$(OBJ)/dbg/%.d)
OBJ_FILES:=$(CPP_FILES:%.cpp=$(OBJ)/opt/%.o)
DBG_FILES:=$(CPP_FILES:%.cpp=$(OBJ)/dbg/%.o)
DBG_OUT:=$(OUT)_dbg

run_dbg: dbg
	$(BIN)/$(DBG_OUT)
run_opt: opt
	$(BIN)/$(OUT)
dbg: $(BIN)/$(DBG_OUT)
opt: $(BIN)/$(OUT)
clean:
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*

$(BIN)/$(OUT): $(BIN) $(OBJ) $(OBJ_FILES)
	$(CXX) $(OBJ_FILES) $(CFLAGS_OPT) $(LFLAGS_OPT) -o $@

$(BIN)/$(DBG_OUT): $(BIN) $(OBJ) $(DBG_FILES)
	$(CXX) $(DBG_FILES) $(CFLAGS_DBG) $(LFLAGS_DBG) -o $@

$(OBJ)/opt/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS_OPT) -c -o $@ -MMD

$(OBJ)/dbg/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS_DBG) -c -o $@ -MMD

$(BIN):
	mkdir -p $@

$(OBJ):
	mkdir -p $@

-include $(DEP_FILES)

.PHONY: run_dbg run_opt dbg opt clean
