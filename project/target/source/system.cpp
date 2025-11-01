//! \note Bra modifikation av system-klassen. Jag har lite kommentarer om kvarvarande kod från den
//!       gamla implementationen, duplicerad kod för temperaturprediktion samt hardkodat pin-nummer
//!       för temperatursensorn. I övrigt ser det stabilt ut.
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

//! \note Denna del med LED states och EEPROM kan tas bort.
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

    //!\note Jag hade tagit bort LED-state + EEPROM grejser om jag var du; det kan dock fungera
    //!      om EEPROM-biten = 1 kommer din predict-timer att starta. Jag hade bara startar
    //!      predictTimer geom att skriva myPredict.start() här i stället.
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
    //! \note Dessa fyra rader är identiska med de fyra rader som används för att prediktera 
    //!       temperaturen i handleButtonPressed(). Om jag var du hade jag skapar en privat
    //!       metod döpt exempelvis predictTemperature() eller dylikt och kallat på denna i stället.
    //!       Nu rör det sig bara om fyra rader, så det är fine. 
    //!
    //!       En till sak - temperaturesensorns pin-nummer är hardkodat, vilket är fine i detta fall.
    //!       Ett förbättringsförslag är dock att du låter använder skicka med temperatursensorns
    //!       pin-nummer till konstruktorn, exempelvis via ytterligare ett ingående argument
    //!       const uint8_t tempSensorPin. Detta pin-nummer hade också kunnat sparas i en
    //!       medlemsvariabel const uint8_t myTempSensorPin, så kan man enkelt byta pin-nummer.
    //!       Detta är dock inget som krävs i uppgiften, men det är bra att tänka på.
    //! 
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

    //! \note Samma kommentar som ovan.
    const auto inputVoltage{myAdc.inputVoltage(2U)};
    const auto mV{inputVoltage * 1000.0};
    const auto temp{myPredict.predict(inputVoltage)};
 
    mySerial.printf("Real input voltage: %d mV, predicted temperature: %d C!\n",  round(mV), round(temp));
    myPredictTimer.restart();
}

// -----------------------------------------------------------------------------
//! \note Denna metod är en kvarleva från den gamla implementationen och kan tas bort.
void System::checkLedStateInEeprom() noexcept
{
    if (readLedStateFromEeprom())
    {
        myPredictTimer.start();
        mySerial.printf("Toggle timer enabled!\n");
    }
}

// -----------------------------------------------------------------------------
//! \note Denna metod är en kvarleva från den gamla implementationen och kan tas bort.
void System::writeLedStateToEeprom() noexcept
{ 
    myEeprom.write(LedState::address, myPredictTimer.isEnabled());
}

// -----------------------------------------------------------------------------
//! \note Denna metod är en kvarleva från den gamla implementationen och kan tas bort.
bool System::readLedStateFromEeprom() const noexcept
{
    uint8_t state{};
    return myEeprom.read(LedState::address, state) ? LedState::enabled == state : false;
}
} // namespace target