# Makefile for LFM3-1.7B Native C++17 Tri-Hybrid Inference Engine (lfm3_v2)
CROSS_COMPILE ?=
CXX            = $(CROSS_COMPILE)g++

RUNTIME_DIR    = ../eswin_runtime
CXXFLAGS      += -std=c++17 -O3 -ffast-math -funroll-loops -fopenmp -pthread -Wall -Wextra \
                 -I$(RUNTIME_DIR)/include -I$(RUNTIME_DIR)/src
LDFLAGS       += -L$(RUNTIME_DIR)/build -les_runtime -fopenmp -ldl -pthread \
                 -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../eswin_runtime/build:/tmp'

TARGET         = lfm3_v2

.PHONY: all clean

all: $(TARGET)

$(TARGET): lfm3_v2.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f $(TARGET)
