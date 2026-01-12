/**
 * @brief Implementation details of the Raspberry Pi LED driver.
 */
#include <cstdint>

#include "driver/gpio/rpi.h"
#include "driver/led/rpi.h"

using namespace driver::gpio;

namespace driver::led
{
// -----------------------------------------------------------------------------
Rpi::Rpi(const std::uint8_t pin, const bool startValue) noexcept
    : myLine{rpi::line::create(pin, rpi::Direction::Output)} 
{ 
    write(startValue); 
}

// -----------------------------------------------------------------------------
Rpi::~Rpi() noexcept { rpi::line::release(myLine); }

// -----------------------------------------------------------------------------
bool Rpi::isInitialized() const noexcept { return nullptr != myLine; }

// -----------------------------------------------------------------------------
bool Rpi::isEnabled() const noexcept { return rpi::line::read(myLine); }

// -----------------------------------------------------------------------------
void Rpi::write(const bool value) noexcept { rpi::line::write(myLine, value); }

// -----------------------------------------------------------------------------
void Rpi::toggle() noexcept { rpi::line::toggle(myLine); }

} // namespace driver::led