# Makefile for Qwen2-v2 Chat Engine
CROSS_COMPILE ?= 
CXX            = $(CROSS_COMPILE)g++

RUNTIME_DIR    = ../eswin_runtime
CXXFLAGS      += -std=c++17 -O2 -Wall -Wextra -pthread -I$(RUNTIME_DIR)/include
LDFLAGS       += -L$(RUNTIME_DIR)/build -les_runtime -ldl -pthread -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../eswin_runtime/build'

TARGET         = qwen2_v2

.PHONY: all clean

all: $(TARGET)

$(TARGET): qwen2_v2.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f $(TARGET)
