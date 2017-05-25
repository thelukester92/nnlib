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

CXX    := g++
CFLAGS := -Wall -std=c++11 -DACCELERATE_BLAS
LFLAGS :=
PREFIX := /usr/local/include

override BIN := bin
override INC := include
override OUT := nnlib_test
CFLAGS += -I$(INC)

override INSTALL_FILES := $(shell find $(INC) -type f)
override INSTALL_FILES := $(INSTALL_FILES:$(INC)/%.h=$(PREFIX)/%.h)

override UNAME := $(shell uname -s)
override GNU   := $(shell $(CXX) --version 2>/dev/null | grep ^g++ | sed 's/^.* //g')
override BLAS  := $(findstring -DACCELERATE_BLAS,$(CFLAGS))

# Link BLAS if applicable
ifneq ($(BLAS),)
ifeq ($(UNAME),Darwin)
	# OSX uses the Accelerate framework
	LFLAGS += -framework Accelerate
ifneq ($(GNU),)
	# gcc on OS X requires this flag for the Accelerate framework to work
	CFLAGS += -flax-vector-conversions
endif
else
	# Other systems use OpenBLAS
	LFLAGS += -L/usr/local/lib -lopenblas -lpthread
endif
endif

# Targets follow

test: $(BIN)/$(OUT)
	$(BIN)/$(OUT)
clean:
	rm -f $(BIN)/$(OUT)
install: $(INSTALL_FILES)
uninstall:
	rm -f $(INSTALL_FILES)

$(BIN)/$(OUT): $(BIN)
	$(CXX) test/main.cpp $(CFLAGS) $(LFLAGS) -o $@

$(PREFIX)/%.h: $(INC)/%.h
	mkdir -p $(dir $@)
	cp $< $@

$(BIN):
	mkdir -p $@

.PHONY: test clean install uninstall
