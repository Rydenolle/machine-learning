/**
 * @brief Raspberry Pi LED driver.
 */
#pragma once

#include <cstdint>

#include "driver/led/interface.h"

/** GPIO line structure. */
struct gpiod_line;

namespace driver::led
{
/**
 * @brief Raspberry Pi LED driver.
 * 
 *        This class is non-copyable and non-movable.
 */
class Rpi final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] pin GPIO pin the LED is connected to.
     * @param[in] startValue Initial value of the LED (default = off).
     */
    explicit Rpi(const std::uint8_t pin, const bool startValue = false) noexcept;

    /**
     * @brief Destructor.
     */
    ~Rpi() noexcept override;

     /**
     * @brief Check whether the LED has been initialized.
     * 
     * @return True if the LED has been initialized, false otherwise.
     */
    bool isInitialized() const noexcept override;

    /**
     * @brief Check whether the LED is enabled/on.
     * 
     * @return True if the LED is enabled, false otherwise.
     */
    bool isEnabled() const noexcept override;

    /**
     * @brief Write output value to enable/disable the LED.
     * 
     * @param value The value to write.
     */
    void write(const bool value) noexcept override;

    /**
     * @brief Toggle the output of the LED.
     */
    void toggle() noexcept override;

    Rpi()                      = delete; // No default constructor.
    Rpi(const Rpi&)            = delete; // No copy constructor.
    Rpi(Rpi&&)                 = delete; // No move constructor.
    Rpi& operator=(const Rpi&) = delete; // No copy assignment.
    Rpi& operator=(Rpi&&)      = delete; // No move assignment.

private:
    /** Pointer to the associated GPIO line. */
    struct gpiod_line* myLine;
};
} // namespace driver::led