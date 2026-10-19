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
                 $(SRC_DIR)/backend/worker_pool.cpp \
                 $(SRC_DIR)/model/sha256.cpp \
                 $(SRC_DIR)/model/model_manifest.cpp \
                 $(SRC_DIR)/model/downloader.cpp \
                 $(SRC_DIR)/api/es_runtime.cpp

OBJS           = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRCS))
TARGET_LIB     = $(BUILD_DIR)/libes_runtime.so
TEST_DMA       = $(BUILD_DIR)/test_dma_alloc
TEST_CONCUR    = $(BUILD_DIR)/test_dual_die_concurrency
TEST_MANIFEST  = $(BUILD_DIR)/test_manifest
TOOL_FETCH     = $(BUILD_DIR)/eswin-fetch

.PHONY: all test clean

all: $(TARGET_LIB) $(TEST_DMA) $(TEST_CONCUR) $(TEST_MANIFEST) $(TOOL_FETCH)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_LIB): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

$(TEST_DMA): $(TEST_DIR)/test_dma_alloc.cpp $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "Built: $@"

$(TEST_CONCUR): $(TEST_DIR)/test_dual_die_concurrency.cpp $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "Built: $@"

$(TEST_MANIFEST): $(TEST_DIR)/test_manifest.cpp $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "Built: $@"

$(TOOL_FETCH): tools/eswin_fetch.cpp $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "Built: $@"

test: $(TEST_DMA) $(TEST_CONCUR) $(TEST_MANIFEST)
	@echo "Running Milestone 1 DMA test suite..."
	@LD_LIBRARY_PATH=$(BUILD_DIR):/lib:/usr/lib:$$LD_LIBRARY_PATH $(TEST_DMA)
	@echo "\nRunning Milestone 2 Dual-Die Concurrency test suite..."
	@LD_LIBRARY_PATH=$(BUILD_DIR):/lib:/usr/lib:$$LD_LIBRARY_PATH $(TEST_CONCUR) 20
	@echo "\nRunning Milestone 3 Model Package Manifest test suite..."
	@LD_LIBRARY_PATH=$(BUILD_DIR):/lib:/usr/lib:$$LD_LIBRARY_PATH $(TEST_MANIFEST)

clean:
	rm -rf $(BUILD_DIR)
