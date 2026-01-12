/**
 * @brief LED driver stub.
 */
#pragma once

#include <cstdint>

#include "driver/led/interface.h"

namespace driver::led
{
/**
 * @brief LED driver stub.
 * 
 *        This class is non-copyable and non-movable.
 */
class Stub final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] pin GPIO pin the LED is connected to.
     * @param[in] startValue Initial value of the LED (default = off).
     */
    explicit Stub(const std::uint8_t pin, const bool startValue = false) noexcept
        : myOutput{startValue}
    {
        // Ignore the pin, as it's not needed in this implementation.
        (void) (pin);
    }

    /**
     * @brief Destructor.
     */
    ~Stub() noexcept override = default;

     /**
     * @brief Check whether the LED has been initialized.
     * 
     * @return True if the LED has been initialized, false otherwise.
     */
    bool isInitialized() const noexcept override { return true; }

    /**
     * @brief Check whether the LED is enabled/on.
     * 
     * @return True if the LED is enabled, false otherwise.
     */
    bool isEnabled() const noexcept override { return myOutput; }

    /**
     * @brief Write output value to enable/disable the LED.
     * 
     * @param value The value to write.
     */
    void write(const bool value) noexcept override { myOutput = value; }

    /**
     * @brief Toggle the output of the LED.
     */
    void toggle() noexcept override { myOutput = !myOutput; }

    Stub()                       = delete; // No default constructor.
    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.

private:
    /** Output value. */
    bool myOutput;
};
} // namespace driver::led