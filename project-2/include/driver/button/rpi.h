/**
 * @brief Raspberry Pi button driver.
 */
#pragma once

#include <cstdint>

#include "driver/button/interface.h"

/** GPIO line structure. */
struct gpiod_line;

namespace driver::button
{
/**
 * @brief Raspberry Pi button driver.
 * 
 *        This class is non-copyable and non-movable.
 */
class Rpi final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] pin GPIO pin the button is connected to.
     */
    explicit Rpi(const std::uint8_t pin) noexcept;

    /**
     * @brief Destructor.
     */
    ~Rpi() noexcept override;

    /**
     * @brief Check whether the button has been initialized.
     * 
     * @return True if the button has been initialized, false otherwise.
     */
    bool isInitialized() const noexcept override;

     /**
     * @brief Check whether the button is pressed.
     * 
     * @return True if the button is pressed, false otherwise.
     */
    bool isPressed() noexcept override;

    /**
     * @brief Check whether a given event has occurred.
     * 
     * @param[in] edge The edge to detect.
     * 
     * @return True if the given event has occurred, false otherwise.
     */
    bool hasEventOccurred(const Edge edge) noexcept override;

    Rpi()                      = delete; // No default constructor.
    Rpi(const Rpi&)            = delete; // No copy constructor.
    Rpi(Rpi&&)                 = delete; // No move constructor.
    Rpi& operator=(const Rpi&) = delete; // No copy assignment.
    Rpi& operator=(Rpi&&)      = delete; // No move assignment.

private:
    /** Pointer to the associated GPIO line. */
    gpiod_line* myLine;

    /** Previous input value. */
    bool myPrevInput;
};
} // namespace rpi