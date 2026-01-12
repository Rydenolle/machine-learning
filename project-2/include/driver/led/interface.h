/**
 * @brief LED interface.
 */
#pragma once

#include <cstdint>

namespace driver::led
{
/**
 * @brief LED interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Check whether the LED has been initialized.
     * 
     * @return True if the LED has been initialized, false otherwise.
     */
    virtual bool isInitialized() const noexcept = 0;

    /**
     * @brief Check whether the LED is enabled/on.
     * 
     * @return True if the LED is enabled, false otherwise.
     */
    virtual bool isEnabled() const noexcept = 0;

    /**
     * @brief Write output value to enable/disable the LED.
     * 
     * @param value The value to write.
     */
    virtual void write(const bool value) noexcept = 0;

    /**
     * @brief Toggle the output of the LED.
     */
    virtual void toggle() noexcept = 0;
};
} // namespace driver::led
