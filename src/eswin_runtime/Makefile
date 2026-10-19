# Makefile for Open ESWIN NPU Runtime (eswin_runtime)

CROSS_COMPILE ?= 
CXX            = $(CROSS_COMPILE)g++
CC             = $(CROSS_COMPILE)gcc

CXXFLAGS      += -std=c++17 -O2 -Wall -Wextra -fPIC -pthread -Iinclude
LDFLAGS       += -ldl -pthread

BUILD_DIR      = build
SRC_DIR        = src
TEST_DIR       = tests

SRCS           = $(SRC_DIR)/backend/driver_wrapper.cpp \
                 $(SRC_DIR)/backend/dma_buffer.cpp \
                 $(SRC_DIR)/api/es_runtime.cpp

OBJS           = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRCS))
TARGET_LIB     = $(BUILD_DIR)/libes_runtime.so
TEST_BIN       = $(BUILD_DIR)/test_dma_alloc

.PHONY: all test clean

all: $(TARGET_LIB) $(TEST_BIN)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_LIB): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

$(TEST_BIN): $(TEST_DIR)/test_dma_alloc.cpp $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "Built: $@"

test: $(TEST_BIN)
	@echo "Running Milestone 1 DMA test suite..."
	@LD_LIBRARY_PATH=$(BUILD_DIR):/lib:/usr/lib:$$LD_LIBRARY_PATH $(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
