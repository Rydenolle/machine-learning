# Application target.
TARGET := gpiod

# C++ compiler.
CXX_COMPILER := g++

# Source files.
SOURCE_FILES := source/driver/button/rpi.cpp \
                source/driver/gpio/rpi.cpp \
			    source/driver/led/rpi.cpp \
			    source/main.cpp \
				source/dense_layer/dense_layer.cpp\
				source/neural_network/single_layer.cpp\

# Main include directory.
INC_DIR := include/

# Linked libraries.
LINK_LIBS := -lgpiod

# Compiler flags.
# Add flag -DRPI when compiling for Raspberry Pi.
COMPILER_FLAGS := -Wall -Werror -std=c++17 -I$(INC_DIR) $(LINK_LIBS)

# Build and run the application as default.
default: build run

# Build the application.
build:
	@$(CXX_COMPILER) $(SOURCE_FILES) -o $(TARGET) $(COMPILER_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)