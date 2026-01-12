/**
 * @brief Raspberry Pi GPIO driver.
 */
#pragma once

#include <cstdint>

/** GPIO line structure. */
struct gpiod_line;

namespace driver::gpio::rpi
{
/**
 * @brief Enumeration of data directions.
 */
enum class Direction : std::uint8_t
{ 
    Input,  ///< Input line.
    Output, ///< Output line.
};

/**
 * @brief Enumeration of edges for event detection.
 */
enum class Edge : std::uint8_t 
{ 
    Rising,  ///< Rising edge (0 -> 1).
    Falling, ///< Falling edge (1 -> 0).
    Both,    ///< Both edges (0 -> 1 or 1 -> 0).
};

namespace line
{
/**
 * @brief Create a new GPIO line for a device.
 * 
 * @param[in] pin GPIO pin the device is connected to.
 * @param[in] direction Data direction of the device.
 * 
 * @return Pointer to the new GPIO line, or nullptr on failure.
 */
gpiod_line* create(const std::uint8_t pin, const Direction direction) noexcept;

/**
 * @brief Release the given line.
 * 
 * @param[in] self Pointer to the GPIO line.
 */
void release(gpiod_line* self) noexcept;

/**
 * @brief Set output of the given line.
 * 
 * @param[in] self Pointer to the GPIO line.
 * @param[in] value The value to set.
 */
void write(gpiod_line* self, const bool value) noexcept;

/**
 * @brief Read the input of the given line.
 * 
 * @param[in] self Pointer to the GPIO line.
 * 
 * @return True of the input is high, false otherwise.
 */
bool read(gpiod_line* self) noexcept;

/**
 * @brief Toggle the output of given GPIO line.
 * 
 * @param[in] self Pointer to the GPIO line.
 */
void toggle(gpiod_line* self) noexcept;

} // namespace line
} // namespace driver::gpio::rpi