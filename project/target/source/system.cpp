/**
 * @brief Generic system implementation details for an MCU with configurable hardware devices.
 */
#include <stdint.h>

#include "driver/adc/interface.h"
#include "driver/eeprom/interface.h"
#include "driver/gpio/interface.h"
#include "driver/serial/interface.h"
#include "driver/timer/interface.h"
#include "driver/watchdog/interface.h"
#include "ml/lin_reg/interface.h"
#include "target/system.h"

namespace target
{
namespace
{
constexpr int round(const double number) noexcept
{
    return 0.0 <= number ? static_cast<int>(number + 0.5) : static_cast<int>(number - 0.5);
}
} // namespace

/**
 * @brief Structure of LED state parameters.
 */
namespace LedState
{
    /** LED state address in EEPROM. */
    static constexpr uint8_t address{0U};

    /** Enabled state value in EEPROM. */
    static constexpr uint8_t enabled{1U};
};

// -----------------------------------------------------------------------------
System::System(driver::GpioInterface& led, driver::GpioInterface& button,
               driver::TimerInterface& debounceTimer, driver::TimerInterface& predictTimer,
               driver::SerialInterface& serial, driver::WatchdogInterface& watchdog,
               driver::EepromInterface& eeprom, driver::AdcInterface& adc, ml::lin_reg::Interface& predict) noexcept
    : myLed{led}
    , myButton{button}
    , myDebounceTimer{debounceTimer}
    , myPredictTimer{predictTimer}
    , mySerial{serial}
    , myWatchdog{watchdog}
    , myEeprom{eeprom}
    , myAdc{adc}
    , myPredict{predict}
{
    myButton.enableInterrupt(true);
    mySerial.setEnabled(true);
    myWatchdog.setEnabled(true);
    myEeprom.setEnabled(true);
    checkLedStateInEeprom();
}

// -----------------------------------------------------------------------------
System::~System() noexcept
{
    myLed.write(false);
    myButton.enableInterrupt(false);
    myDebounceTimer.stop();
    myPredictTimer.stop();
    myWatchdog.setEnabled(false);
}

// -----------------------------------------------------------------------------
void System::enableSerialTransmission(const bool enable) noexcept
{
    mySerial.setEnabled(enable);
}

// -----------------------------------------------------------------------------
void System::handleButtonInterrupt() noexcept
{
    myButton.enableInterruptOnPort(false);
    myDebounceTimer.start();
    if (myButton.read()) { handleButtonPressed(); }
}

// -----------------------------------------------------------------------------
void System::handleDebounceTimerInterrupt() noexcept
{
    myDebounceTimer.stop();
    myButton.enableInterruptOnPort(true);
}

// -----------------------------------------------------------------------------
void System::handlepredictTimerInterrupt() noexcept 
{ 
    // mySerial.printf("Toggling the LED!\n");
    // myLed.toggle(); 

    // Prediktera temperature i stället, skit i LEDen.
    const auto inputVoltage{myAdc.inputVoltage(2U)};
    const auto mV{inputVoltage * 1000.0};
    const auto temp{myPredict.predict(inputVoltage)};
 
    mySerial.printf("Real input voltage: %d mV, predicted temperature: %d C!\n",  round(mV), round(temp));
}

// -----------------------------------------------------------------------------
void System::run() noexcept
{
    mySerial.printf("Running the system!\n");
    
    while (1)
    {
        myWatchdog.reset();
    }
}

// -----------------------------------------------------------------------------
void System::handleButtonPressed() noexcept
{
    mySerial.printf("Button pressed!\n");
    // myPredictTimer.toggle();
    // writeLedStateToEeprom();

    const auto inputVoltage{myAdc.inputVoltage(2U)};
    const auto mV{inputVoltage * 1000.0};
    const auto temp{myPredict.predict(inputVoltage)};
 
    mySerial.printf("Real input voltage: %d mV, predicted temperature: %d C!\n",  round(mV), round(temp));

    myPredictTimer.restart();

    // Återstarta 60-sekunderstimern.

/*     if (myPredictTimer.isEnabled()) { mySerial.printf("Toggle timer enabled!\n"); }
    else
    {
        mySerial.printf("Toggle timer disabled!\n");
        myLed.write(false);
    } */
}

// -----------------------------------------------------------------------------
void System::checkLedStateInEeprom() noexcept
{
    if (readLedStateFromEeprom())
    {
        myPredictTimer.start();
        mySerial.printf("Toggle timer enabled!\n");
    }
}

// -----------------------------------------------------------------------------
void System::writeLedStateToEeprom() noexcept
{ 
    myEeprom.write(LedState::address, myPredictTimer.isEnabled());
}

// -----------------------------------------------------------------------------
bool System::readLedStateFromEeprom() const noexcept
{
    uint8_t state{};
    return myEeprom.read(LedState::address, state) ? LedState::enabled == state : false;
}
} // namespace target