/**
 * @brief Button interface.
 */
#pragma once

#include <cstdint>

namespace driver::button
{
/**
 * @brief Enumeration of button events.
 */
enum class Edge : std::uint8_t
{ 
    Rising,  ///< Rising edge (0 -> 1).
    Falling, ///< Falling edge (1 -> 0).
    Both,    ///< Both edges (0 -> 1 or 1 -> 0).
};

/**
 * @brief Button interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Check whether the button has been initialized.
     * 
     * @return True if the button has been initialized, false otherwise.
     */
    virtual bool isInitialized() const noexcept = 0;

    /**
     * @brief Check whether the button is pressed.
     * 
     * @return True if the button is pressed, false otherwise.
     */
    virtual bool isPressed() noexcept = 0;

    /**
     * @brief Check whether a given event has occurred.
     * 
     * @param[in] edge The edge to detect.
     * 
     * @return True if the given event has occurred, false otherwise.
     */
    virtual bool hasEventOccurred(const Edge edge) noexcept = 0;
};
} // namespace driver::button
