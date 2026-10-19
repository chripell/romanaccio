CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall -Wextra
LDFLAGS ?= -fopenmp -lm -ldl

TARGET = smollm_chat

all: $(TARGET)

$(TARGET): smollm_chat.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
