/**
 * @brief Demonstration of GPIO device drivers in C++:
 * 
 *        The following devices are used:
 *            - A button connected to pin 13 on the device toggles a timer.
 *            - The aforementioned timer toggles an LED every 100 ms when enabled.
 *            - Another timer reduces the effect of contact bounces after pushing the button.
 *            - A watchdog timer is used to restart the program if it gets stuck somewhere.
 *            - An EEPROM stream is used to store the LED state. On startup, this value is read;
 *              if the last stored state before power down was "on," the LED will automatically blink.
 */
#include "driver/atmega328p/adc.h"
#include "driver/atmega328p/eeprom.h"
#include "driver/atmega328p/gpio.h"
#include "driver/atmega328p/serial.h"
#include "driver/atmega328p/timer.h"
#include "driver/atmega328p/watchdog.h"
#include "ml/lin_reg/lin_reg.h"
#include "target/system.h"

using namespace driver::atmega328p;

namespace
{
/** Pointer to the system implementation. */
target::System* mySys{nullptr};

/**
 * @brief Callback for the button.
 */
void buttonCallback() noexcept { mySys->handleButtonInterrupt(); }

/**
 * @brief Callback for the debounce timer.
 * 
 *        This callback is invoked whenever the debounce timer elapses.
 */
void debounceTimerCallback() noexcept { mySys->handleDebounceTimerInterrupt(); }

/**
 * @brief Callback for the toggle timer.
 * 
 *        This callback is invoked whenever the toggle timer elapses.
 */
void predictTimerCallback() noexcept { mySys->handlepredictTimerInterrupt(); }

/**
 * @brief Round a double to nearest integer to be able to print the number.
 * 
 * @param[in] number Double number to be rounded, positive or negative.
 * 
 * @return Rounded double as an integer.
 */
constexpr int round(const double number) noexcept
{
    return 0.0 <= number ? static_cast<int>(number + 0.5) : static_cast<int>(number - 0.5);
}

} // namespace

/**
 * @brief Initialize and run the system on the target MCU.
 * 
 * @return 0 on termination of the program (should never occur).
 */
int main()
{
    // Obtain a reference to the singleton serial device instance.
    auto& serial{Serial::getInstance()};
    serial.setEnabled(true);

    // Vector containing training data. Needs to be 10-15 sets of data. 
    const container::Vector<double> trainInput{0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 
                                               2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9};
    const container::Vector<double> trainOutput{-50.0, -20.0, 10.0, 40.0, 70.0, 100.0, 130.0, 
                                                160.0, 190.0, 220.0, 250.0, 280.0, 310.0,};

    // 
    ml::lin_reg::LinReg linRegModel{trainInput, trainOutput};

    // Train the linear regression model on the training data.
    if (linRegModel.train(1000, 0.1))
    {
        for (const auto& x : trainInput)
        {
            const auto voltage_mV{x * 1000.0};
            serial.printf("x = %d mV, yref = %d C\n", round(voltage_mV), round(linRegModel.predict(x)));
        }
    }
    else
    {
        serial.printf("Training failed.");
        return -1;
    }

    // Obtain a reference to the singleton ADC instance.
    auto& adc{Adc::getInstance()};
    adc.setEnabled(true);

    // Initialize the GPIO devices.
    Gpio led{8U, Gpio::Direction::Output};
    Gpio button{13U, Gpio::Direction::InputPullup, buttonCallback};

    // Initialize the timers.
    Timer debounceTimer{300U, debounceTimerCallback};
    Timer predictTimer{60000UL, predictTimerCallback};

    // Start the 60 second predict timer.
    predictTimer.start();

    // Obtain a reference to the singleton watchdog timer instance.
    auto& watchdog{Watchdog::getInstance()};

    // Obtain a reference to the singleton EEPROM instance.
    auto& eeprom{Eeprom::getInstance()};

    // Initialize the system with the given hardware.
    target::System system{led, button, debounceTimer, predictTimer, serial, watchdog, eeprom, adc, linRegModel};
    mySys = &system;

    // Run the system perpetually on the target MCU.
    mySys->run();

    // This point should never be reached; the system is intended to run indefinitely on the target MCU.
    return 0;
}
