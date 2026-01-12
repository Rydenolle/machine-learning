/**
 * @brief Button driver implementation details for Raspberry Pi.
 */
#include <chrono>
#include <cstdint>
#include <thread>

#include "driver/button/rpi.h"
#include "driver/gpio/rpi.h"

using namespace driver::gpio;

namespace driver::button
{
namespace
{
// -----------------------------------------------------------------------------
bool detectEvent(gpiod_line* self, const Edge edge, bool& prevInput) noexcept
{
    if (nullptr == self) { return false; }

    // Wait for 50 ms to mitigate effects of debounces.
    constexpr std::uint8_t waitTime_ms{50U};
    std::this_thread::sleep_for(std::chrono::milliseconds(waitTime_ms));

    // Read the new input value, store away the previous one.
    const auto oldVal{prevInput};
    const auto newVal{rpi::line::read(self)};
    prevInput = newVal;

    // Return true if given event has occurred.
    switch (edge)
    {
        case Edge::Rising:
            return newVal && !oldVal;
        case Edge::Falling:
            return !newVal && oldVal;
        default:
            return newVal != oldVal;
    }
}
} // namespace

// -----------------------------------------------------------------------------
Rpi::Rpi(const std::uint8_t pin) noexcept
    : myLine{rpi::line::create(pin, rpi::Direction::Input)}
    , myPrevInput{false}
{}

// -----------------------------------------------------------------------------
Rpi::~Rpi() noexcept 
{
    rpi::line::release(myLine);
}

// -----------------------------------------------------------------------------
bool Rpi::isInitialized() const noexcept { return nullptr != myLine; }

// -----------------------------------------------------------------------------
bool Rpi::isPressed() noexcept
{ 
    // Store the current value as previous input for future event detection.
    myPrevInput = rpi::line::read(myLine);
    return myPrevInput;
}

// -----------------------------------------------------------------------------
bool Rpi::hasEventOccurred(const Edge edge) noexcept
{
    // Compare the current input with the previous value stored via 'isPressed()'.
    // Update the previous value via 'detectEvent()'.
    // Return true if the specified event has occured.
    return detectEvent(myLine, edge, myPrevInput);
}
} // namespace driver::button
