/**
 * @brief Neural network implementation with dense layers.
 */
#include <iomanip>
#include <iostream>

#include "ml/dense_layer/dense_layer.h"
#include "ml/neural_network/single_layer.h"

// For testing.
/*
#include <chrono>
#include <thread>

#include "driver/button/stub.h"
#include "driver/led/stub.h"
*/

// For "real" use.
#include "driver/button/rpi.h"
#include "driver/led/rpi.h"

namespace
{
/**
 * @brief Print numbers held by the given vector.
 *
 * @param[in] numbers Vector holding the numbers to print.
 * @param[in] ostream Output stream to use (default = terminal print).
 */
void printNumbers(const std::vector<double>& numbers, std::ostream& ostream = std::cout)
{
    // Terminate the function if the vector is empty.
    if (numbers.empty()) { return; }

    // Get the address of the last element to not print a blank space after the last number.
    const auto end{&numbers[numbers.size() - 1U]};

    // Print each number on a single line, separated by a blank space.
    for (const auto& number : numbers)
    {
        ostream << number;
        if (end != &number) { ostream << " "; }
    }
}

/**
 * @brief Perform prediction with the given neural network, 1 decimal precision.
 *
 * @param[in] network Neural network to predict with.
 * @param[in] inputData Input data to predict with.
 * @param[in] ostream Output stream to use (default = terminal print).
 */
void predict(ml::neural_network::Interface& network,
             const std::vector<std::vector<double>>& inputData,
             std::ostream& ostream = std::cout)
{
    ostream << std::fixed << std::setprecision(1);

    // Terminate the function if no input data is provided.
    if (inputData.empty())
    {
        ostream << "No input data!\n";
        return;
    }
    ostream << "--------------------------------------------------------------------------------\n";

    // Perform prediction with each input value, print the result in the terminal.
    for (const auto& input : inputData)
    {
        const auto prediction{network.predict(input)};
        ostream << "Input: ";
        printNumbers(input, ostream);
        ostream << ", predicted output: ";
        printNumbers(prediction, ostream);
        ostream << "\n";
    }
    ostream << "--------------------------------------------------------------------------------\n\n";
}


} // namespace

/**
 * @brief Train and predict with a simple neural network.
 *
 * @return 0 on success, or -1 on failure.
 */
int main()
{
    // Implement pin numbers for one LED and four buttons.
    constexpr std::uint8_t ledPin{21U};
    constexpr std::uint8_t buttonPin0{1U};
    constexpr std::uint8_t buttonPin1{7U};
    constexpr std::uint8_t buttonPin2{8U};
    constexpr std::uint8_t buttonPin3{25U};

    //! @todo (stämmer detta?) Implement the total number of buttons, for use in a simulation loop.
    constexpr std::size_t buttonCount{4U};

    // Implement the neural network parameters as compile-time constants.
    constexpr std::size_t inputCount{4U};
    constexpr std::size_t hiddenCount{5U};
    constexpr std::size_t outputCount{1U};

    // Implement the number of epochs to train and the learning rate as compile-time constants.
    constexpr std::size_t epochCount{20000};
    constexpr double learningRate{0.01};
    constexpr double targetPrecision{0.99};

    // Create training data vectors.
    const std::vector<std::vector<double>> trainInput{{0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1}
                                                    , {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1}
                                                    , {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1}
                                                    , {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}};
    const std::vector<std::vector<double>> trainOutput{{0}, {1}, {1}, {0}
                                                    ,  {1}, {0}, {0}, {1}
                                                    ,  {1}, {0}, {0}, {1}
                                                    ,  {0}, {1}, {1}, {0}};

    // Create the dense layers.
    ml::dense_layer::DenseLayer hiddenLayer{hiddenCount, inputCount, ml::ActFunc::Tanh};
    ml::dense_layer::DenseLayer outputLayer{outputCount, hiddenCount};

    // Create a single-layer neural network.
    ml::neural_network::SingleLayer network{hiddenLayer, outputLayer, trainInput, trainOutput};

    // Use stubs if testing.
    /*
    using ledDriver = driver::led::Stub;
    using buttonDriver = driver::button::Stub;
    */

    // Use "real" drivers otherwise.
    using ledDriver = driver::led::Rpi;
    using buttonDriver = driver::button::Rpi;

    // Create RPi or stub LED and buttons.
    ledDriver led{ledPin};
    buttonDriver button0{buttonPin0};
    buttonDriver button1{buttonPin1};
    buttonDriver button2{buttonPin2};
    buttonDriver button3{buttonPin3};

    // Create a vector holding the buttons.
    const std::vector<driver::button::Interface*> buttons{&button0, &button1, &button2, &button3};

    // Implement previous LED state check.
    bool prevState{false};

    while (1)
    {
        constexpr double error{-1.0};
        const auto precision{network.train(epochCount, learningRate)};

        // Terminate the program with error code -1 on failure.
        if (error == precision)
        {
            std::cout << "Training failed!\n";
            return -1;
        }
        std::cout << "Precision: " << std::setprecision(2) <<  precision << "\n";
        if (targetPrecision <= precision) { break; }
    }

    // Perform prediction with the network, then terminate the program.
    predict(network, trainInput);

    // Vector holding button inputs.
    std::vector<double> buttonInputs(buttons.size());

    // Test loop with simulated button presses.
    /*
    for (std::size_t i{}; i < 16; ++i)
    {
        for (std::size_t j{}; j < buttonCount; ++j)
            buttonInputs[j] = (i & (1U << j)) ? 1.0 : 0.0;

        std::vector<double> output{network.predict(buttonInputs)};
        const bool state = (output[0] >= 0.5);

        std::cout << "Button inputs:\n(LSB -> MSB (sorry!))\n";
        printNumbers(buttonInputs);
        std::cout << "\n\n";

        led.write(state);

        if (state != prevState)
        {

            std::cout << "Result:\n";
            std::cout << (state ? ".~~* LED\tON *~~." : "*:.. LED\tOFF ..:*") << "\n\n";

            prevState = state;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    */

    // Continuous loop - check the buttons, control the LED accordingly.
    while (1)
    {
        const bool eventOccured{button0.hasEventOccurred(driver::button::Edge::Both) ||
                                button1.hasEventOccurred(driver::button::Edge::Both) ||
                                button2.hasEventOccurred(driver::button::Edge::Both) ||
                                button3.hasEventOccurred(driver::button::Edge::Both)};

        if (!eventOccured) { continue; }

        // Convert button states to floating-point numbers to feed it to the neural network.
        for (std::size_t i{}; i < buttonCount; ++i)
        {
            // Check the button, store 1.0 on the input vector if pressed, else 0.0.
            buttonInputs[i] = buttons[i]->isPressed() ? 1.0 : 0.0;
        }

        std::vector<double> output{network.predict(buttonInputs)};
        const bool state{output[0] >= 0.5};
        led.write(state);

        if (state != prevState)
        {
            std::cout << "Button inputs:\n(LSB -> MSB (sorry!))\n";
            printNumbers(buttonInputs);
            std::cout << "\n\nResult:\n";
            std::cout << (state ? ".~~* LED\tON *~~." : "*:.. LED\tOFF ..:*") << "\n\n";

            prevState = state;
        }

    }

    return 0;
}