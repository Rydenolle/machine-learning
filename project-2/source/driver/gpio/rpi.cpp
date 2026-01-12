/**
 * @brief Raspberry Pi GPIO driver implementation details.
 */
#include <cstdint>

#include <gpiod.h>

#include "driver/gpio/rpi.h"

namespace driver::gpio::rpi
{
namespace
{
/** GPIO chip path. */
constexpr const char* ChipPath{"/dev/gpiochip0"};

/** GPIO chip instance. */
gpiod_chip* myChip{gpiod_chip_open(ChipPath)};

} // namespace

namespace line
{
// -----------------------------------------------------------------------------
gpiod_line* create(const std::uint8_t pin, const Direction direction)  noexcept
{
    // Create new line instance, return a nullptr on failure.
    gpiod_line* self{gpiod_chip_get_line(myChip, pin)};
    if (nullptr == self) { return nullptr; }

    constexpr const char* alias{""};
    constexpr int startVal{0};

    // Set the data direction.
    if (Direction::Input == direction) { gpiod_line_request_input(self, alias); }
    else { gpiod_line_request_output(self, alias, startVal); }

    // Return a pointer to the new instance.
    return self;
}

// -----------------------------------------------------------------------------
void release(gpiod_line* self) noexcept 
{ 
    if (nullptr != self) { gpiod_line_release(self); }
}

// -----------------------------------------------------------------------------
void write(gpiod_line* self, const bool value) noexcept
{
    if (nullptr != self)
    {
        const int newValue{static_cast<int>(value)}; 
        gpiod_line_set_value(self, newValue); 
    }
}

// -----------------------------------------------------------------------------
bool read(gpiod_line* self) noexcept
{
    constexpr int enabled{1};
    return nullptr != self ? enabled == gpiod_line_get_value(self) : false;
}

// -----------------------------------------------------------------------------
void toggle(gpiod_line* self) noexcept
{
    if (nullptr != self)
    {
        const int newValue{static_cast<int>(!read(self))};
        gpiod_line_set_value(self, newValue);
    }
}
} // namespace line
} // namespace driver::gpio::rpi