/**
 * @brief Button driver stub.
 */
#pragma once

#include <cstdint>

#include "driver/button/interface.h"

namespace driver::button
{
/**
 * @brief Button driver stub.
 * 
 *        This class is non-copyable and non-movable.
 */
class Stub final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] pin GPIO pin the button is connected to.
     */
    explicit Stub(const std::uint8_t pin) noexcept
        : myInput{false}
        , myPrevInput{false}
    {
        // Ignore the pin, as it's not needed in this implementation.
        (void) (pin);
    }

    /**
     * @brief Destructor.
     */
    ~Stub() noexcept override = default;

    /**
     * @brief Check whether the button has been initialized.
     * 
     * @return True if the button has been initialized, false otherwise.
     */
    bool isInitialized() const noexcept override { return true; }

     /**
     * @brief Check whether the button is pressed.
     * 
     * @return True if the button is pressed, false otherwise.
     */
    bool isPressed() noexcept override { return myInput; }

    /**
     * @brief Check whether a given event has occurred.
     * 
     * @param[in] edge The edge to detect.
     * 
     * @return True if the given event has occurred, false otherwise.
     */
    bool hasEventOccurred(const Edge edge) noexcept override
    {
        // Ensure that an event is only detected once.
        const auto prevInput{myPrevInput};
        myPrevInput = myInput;

        // Return true if the given event has occurred.
        switch (edge)
        {
            case Edge::Rising:
                return myInput && !prevInput;
            case Edge::Falling:
                return !myInput && prevInput;
            default:
                return myInput != prevInput;
        }
    }

    /**
     * @brief Simulate a button press or release.
     * 
     * @param[in] input True to simulate pressed, false for released.
     */
    void simulateInput(const bool input) noexcept 
    { 
        myPrevInput = myInput;
        myInput     = input; 
    }

    Stub()                       = delete; // No default constructor.
    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.

private:
    /** Current input value. */
    bool myInput;

    /** Previous input value. */
    bool myPrevInput;
};
} // namespace Stub