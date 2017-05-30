# This Makefile is used to:
#   A) Test nnlib.
#   B) Install nnlib headers.
#
# Targets:
#   test      - build and run all unit tests
#   clean     - remove test executable
#   install   - install all headers
#   uninstall - uninstall all headers
#
# Variables that can be modified:
#   CXX    - compiler to use; defaults to g++ (which is clang++ on OS X)
#   CFLAGS - compile flags for test; -DACCELERATE_BLAS to test BLAS
#   LFLAGS - linker flags for test
#   PREFIX - where to install headers; defaults to /usr/local/include

CXX    ?= g++
CFLAGS := -Wall -DACCELERATE_BLAS
LFLAGS :=
PREFIX := /usr/local/include

override BIN := bin
override OBJ := obj
override INC := include
override TST := test
override OUT := nnlib_test
override CFLAGS += -std=c++11 -I$(INC) --coverage

override INSTALL_FILES := $(shell find $(INC) -type f)
override INSTALL_FILES := $(INSTALL_FILES:$(INC)/%.h=$(PREFIX)/%.h)

override CPP_FILES := $(wildcard $(TST)/*.cpp) $(wildcard $(TST)/**/*.cpp)
override DEP_FILES := $(CPP_FILES:$(TST)/%.cpp=$(OBJ)/%.d)
override OBJ_FILES := $(CPP_FILES:$(TST)/%.cpp=$(OBJ)/%.o)

override UNAME := $(shell uname -s)
override GNU   := $(shell $(CXX) --version 2>/dev/null | grep ^g++ | sed 's/^.* //g')
override BLAS  := $(findstring -DACCELERATE_BLAS,$(CFLAGS))

# Link BLAS if applicable
ifneq ($(BLAS),)
ifeq ($(UNAME),Darwin)
	# OSX uses the Accelerate framework
	override LFLAGS += -framework Accelerate
ifneq ($(GNU),)
	# gcc on OS X requires this flag for the Accelerate framework to work
	override CFLAGS += -flax-vector-conversions
endif
else
	# Other systems use OpenBLAS
	override LFLAGS += -L/usr/local/lib -lopenblas -lpthread
endif
endif

# Targets follow

test: $(BIN)/$(OUT)
	$(BIN)/$(OUT)
clean:
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
install: $(INSTALL_FILES)
uninstall:
	rm -f $(INSTALL_FILES)

$(BIN)/$(OUT): $(BIN) $(OBJ_FILES)
	$(CXX) $(OBJ_FILES) $(CFLAGS) $(LFLAGS) -o $@

$(OBJ)/%.o: $(TST)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -o $@ -MMD

$(PREFIX)/%.h: $(INC)/%.h
	mkdir -p $(dir $@)
	cp $< $@

$(BIN):
	mkdir -p $@

.PHONY: test clean install uninstall

-include $(DEP_FILES)
