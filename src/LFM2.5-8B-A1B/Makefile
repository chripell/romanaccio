# Makefile for LFM2.5-8B-A1B Native C++17 Sparse MoE Dual-Die Inference Engine (lfm2_moe)
CROSS_COMPILE ?=
CXX            = $(CROSS_COMPILE)g++

RUNTIME_DIR    = ../eswin_runtime
CXXFLAGS      += -std=c++17 -O3 -ffast-math -funroll-loops -fopenmp -pthread -Wall -Wextra \
                 -I$(RUNTIME_DIR)/include -I$(RUNTIME_DIR)/src
LDFLAGS       += -L$(RUNTIME_DIR)/build -les_runtime -fopenmp -ldl -pthread \
                 -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../eswin_runtime/build:/tmp'

TARGET         = lfm2_moe

.PHONY: all clean

all: $(TARGET)

$(TARGET): lfm2_moe.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f $(TARGET)
