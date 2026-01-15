# Application target.
TARGET := app

# Source files.
SOURCE_FILES := source/dense_layer/dense_layer.cpp\
                source/driver/button/rpi.cpp \
                source/driver/gpio/rpi.cpp \
			    source/driver/led/rpi.cpp \
                source/main.cpp\
				source/neural_network/single_layer.cpp\

# Include directory.
INCLUDE_DIR := include

# Compiler flags.
COMPILER_FLAGS := -Wall -Werror -std=c++17 -lgpiod

# Build and run the application, then clean as default.
default: build run clean

# Build the application.
build:
	@g++ $(SOURCE_FILES) -o $(TARGET) -I$(INCLUDE_DIR) $(COMPILER_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
