/**
 * @brief Linear regression implementation details.
 */
#include "container/vector.h"
#include "ml/lin_reg/lin_reg.h"
#include "driver/atmega328p/serial.h"

#include <stdlib.h>
#include <time.h>

namespace ml
{
namespace lin_reg
{
namespace
{
//--------------------------------------------------------------------------------
size_t getTrainSetCount(const container::Vector<double>& input, 
                        const container::Vector<double>& output) noexcept
{
    return (input.size() <= output.size()) ? input.size() : output.size();
}

/**
 * @brief Get the absolute value of the given number.
 * 
 * @param[in] num The number in question.
 * 
 * @return The absolute value of the given number.
 */
constexpr double absVal(const double num) noexcept { return 0.0 <= num ? num : -num; }

/**
 * @brief Initialize the random generator (once only).
 */
void initRandom() noexcept
{
    // Create a static local variable, which indicates whether the generator has been initialized.
    // This line (with the initialization) is only run once.
    static auto initialized{false};

    // Terminate the function if the generator already has been initialized.
    if (initialized) { return; }

    // Initialize the generator, use the current time as seed (start of the random sequence).
    // time(nullptr) provides the current time.
    srand(time(nullptr));

    // Mark the random generator as initialized.
    initialized = true;
}

/**
 * @brief Get a random floating-point number between 0.0 - 1.0.
 * 
 * @return Random number between 0.0 - 1.0
 */
double randomStartVal() noexcept
{
    initRandom();
    // Divide rand() / RAND_MAX, beware integer division.
    return rand() / static_cast<double>(RAND_MAX);
}

/**
 * @brief Shuffle the content of the given vector.
 * 
 * @param[in] data Reference to the vector to shuffle.
 */
void shuffle(container::Vector<size_t>& data) noexcept
{
    // Get a random integer between 0 - data.size() - 1, call this index r.
    for (size_t i{}; i < data.size(); ++i)
    {
        // Get a random index r (between 0-4 if we have five training sets).
        const auto r{rand() % data.size()};

        // Swap the elements at index i and r => make a copy of data[i].
        const auto temp{data[i]};

        // Copy data[r] to data[i] => now we have two instances of data[r] in the vector.
        data[i] = data[r];

        // Finally put the copy of the "old" data[i] to data[r] => we have swapped the elements.
        data[r] = temp;
    }
}

// Initialize Serial to be able to use serial monitor prints.
auto& serial{driver::atmega328p::Serial::getInstance()};

} // namespace

//--------------------------------------------------------------------------------
LinReg::LinReg(const container::Vector<double>& trainInput, const container::Vector<double>& trainOutput) noexcept
    : myTrainInput{trainInput}
    , myTrainOutput{trainOutput}
    , myTrainSetCount{getTrainSetCount(trainInput, trainOutput)}
    , myBias{randomStartVal()}
    , myWeight{randomStartVal()}
{} 

//--------------------------------------------------------------------------------
double LinReg::LinReg::predict(const double input) const noexcept
{
    return myWeight * input + myBias;
}

//--------------------------------------------------------------------------------
bool LinReg::LinReg::train(const size_t epochCount, const double learningRate) noexcept
{
    // Return false if epochCount == 0, learningRate <= 0.0 or trainingSetCount == 0.
    if ((0U == epochCount) || (0.0 >= learningRate) || (0U == myTrainSetCount)) { return false; }

    // Create a vector holding indexes of the training sets (for randomizing the training order).
    container::Vector<size_t> trainIndex(myTrainSetCount);
    
    // Add the indexes of the training sets (0, 1, 2... setCount - 1).
    for (size_t i{}; i < myTrainSetCount; ++i) { trainIndex[i] = i; }

    // Train for `epochCount` epochs or until the model predicts within 1e-6 of the yref values.
    for (size_t epoch{}; epoch < epochCount; ++epoch)
    {
        double totalErr{};

        // Shuffle the content of trainIndex, i.e., randomize the order of the training sets.
        
        shuffle(trainIndex);

        // Iterate through the training sets one by one, use the random order from trainIndex.
        for (const auto& i : trainIndex)
        {
            // Get the values of the current training set.
            const auto x{myTrainInput[i]};
            const auto yref{myTrainOutput[i]};

            // If x == 0.0, we just got our bias value, i.e., m = yref.
            if (0.0 == x) { myBias = yref; }

            // Optimize the parameters as usual if x != 0.0.
            else
            {
                // Perform prediction and calculate the error/deviation.
                const auto ypred{predict(x)};
                const auto e{yref - ypred};
                
                // Modify the bias: m = m + e * LR.
                myBias += e * learningRate;

                // Modify the weight: k = k + e * LR * x.
                myWeight += e * learningRate * x;

                // Add the absolute error value to the total.
                totalErr += absVal(e);
            }
        }
        // Calculate the average error of this epoch.
        const auto avgErr{totalErr / myTrainSetCount};
        constexpr auto threshold{1.0 / 1000000};

        // Stop training if the average error is less than one in a million.
        if (threshold > avgErr)
        {
            serial.printf("Training finished after %d epochs!\n", epoch);
            return true;
        }        
    }

    return true;
}
} // namespace lin_reg
} // namespace ml