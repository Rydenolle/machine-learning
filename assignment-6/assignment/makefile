# Application target.
TARGET := main

# C++ compiler.
CXX_COMPILER := g++

# Source files.
SOURCE_FILES := main.cpp \
				# source/conv_layer.cpp \

# Main include directory.
INC_DIR := include/

# Compiler flags
CXX_FLAGS := -std=c++17 -I$(INC_DIR) -Wall -Werror # Ifdef flag here.


# Build, run and clean the application as default.
default: build run clean

# Build the application.
build:
	@$(CXX_COMPILER) $(SOURCE_FILES) -o $(TARGET) $(CXX_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
	@rm -f $(TARGET)