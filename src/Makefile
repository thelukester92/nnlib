BIN:=../bin
OBJ:=../obj
INC:=../include
OUT:=nnlib_test
CFLAGS:=-Wall -std=c++11 -I$(INC)
LFLAGS:=
CXX:=g++
UNAME:=$(shell uname -s)

prefix?=/usr/local
INSTALL_INC:=$(prefix)/include
INSTALL_FLAGS:=-fPIC -shared
INSTALL_INC_FILES:=$(wildcard $(INC)/* $(INC)/**/*.h)
INSTALL_INC_FILES:=$(INSTALL_INC_FILES:$(INC)/%.h=$(INSTALL_INC)/%.h)

ifeq ($(UNAME),Darwin)
	LFLAGS+=-framework Accelerate
	CFLAGS+=-DAPPLE
else
	LFLAGS+=-lblas -llapack
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

dbg: $(BIN)/$(DBG_OUT)
opt: $(BIN)/$(OUT)
install: $(INSTALL_INC_FILES)
clean:
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
uninstall:
	rm -rf $(INSTALL_INC)/nnlib $(INSTALL_INC)/nnlib.h

$(BIN)/$(OUT): $(BIN) $(OBJ) $(OBJ_FILES)
	$(CXX) $(OBJ_FILES) $(CFLAGS_OPT) $(LFLAGS_OPT) -o $@

$(BIN)/$(DBG_OUT): $(BIN) $(OBJ) $(DBG_FILES)
	$(CXX) $(DBG_FILES) $(CFLAGS_DBG) $(LFLAGS_DBG) -o $@

$(INSTALL_INC)/%.h: $(INC)/%.h
	mkdir -p $(dir $@)
	cp $< $@

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

.PHONY: dbg opt install clean uninstall
