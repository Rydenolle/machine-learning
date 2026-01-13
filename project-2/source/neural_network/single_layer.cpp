/**
 * @brief Single Layer implementation details.
 */
#include <vector>

#include "ml/neural_network/single_layer.h"

namespace ml::neural_network
{
namespace
{
/**
 * @brief Method for making sure the training set count is not larger than the smallest value of
 *        either the input or the output.
 *
 * @param[in] input 2D vector containing float numbers.
 * @param[in] output 2D vector containing float numbers.
 *
 * @return The size of the smallest vector as an unsigned integer.
 */
std::size_t getTrainSetCount(const std::vector<std::vector<double>>& input,
                             const std::vector<std::vector<double>>& output) noexcept
{
    return (input.size() <= output.size() ? input.size() : output.size());
}

/**
 * @brief Get the absolute value of the given number.
 *
 * @param[in] num The number in question.
 *
 * @return The absolute value of the given number.
 */
constexpr double absVal(const double num) noexcept { return 0.0 <= num ? num : -num; }

} // namespace

//--------------------------------------------------------------------------------
SingleLayer::SingleLayer(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                         const std::vector<std::vector<double>>& trainInput,
                         const std::vector<std::vector<double>>& trainOutput) noexcept
    : myHiddenLayer(hiddenLayer)
    , myOutputLayer(outputLayer)
    , myTrainInput(trainInput)
    , myTrainOutput(trainOutput)
    , myTrainSetCount(getTrainSetCount(trainInput, trainOutput))
{}

//--------------------------------------------------------------------------------
const std::vector<double>& SingleLayer::predict(const std::vector<double>& input)
{
    myHiddenLayer.feedforward(input);
    myOutputLayer.feedforward(myHiddenLayer.output());
    return myOutputLayer.output();
}

//--------------------------------------------------------------------------------
bool SingleLayer::train(std::size_t epochCount, double learningRate)
{

    // Return false if any faulty epoch count, train set or learning rate.
    if (0U == myTrainSetCount || 0U == epochCount || 0.0 >= learningRate) { return false; }

    //! @todo? Återinitierat dense-lagerparametrarna.
    myHiddenLayer.initParams();
    myOutputLayer.initParams();

    // Train for 'epochCount' number of times.
    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        for (std::size_t i = 0U; i < myTrainSetCount; ++i)
        {
            // Get the values of the current training set.
            const auto& input = myTrainInput[i];
            const auto& output = myTrainOutput[i];

            // Feedforward.
            myHiddenLayer.feedforward(input);
            myOutputLayer.feedforward(myHiddenLayer.output());

            // Backpropagate.
            myOutputLayer.backpropagate(output);
            myHiddenLayer.backpropagate(myOutputLayer);

            // Optimize.
            myOutputLayer.optimize(myHiddenLayer.output(), learningRate);
            myHiddenLayer.optimize(input, learningRate);
        }
    }
    //! @todo Beräkna precisionen här! Ta gärna accuracy-metoderna från referenskoden.
    // return accuracy();

    // i main(?): while train() < 0.99 ?
    return true;

}

//! @todo fixa --------------------------------------------------------------------------------
double SingleLayer::accuracy(/*const std::vector<std::vector<double>>& trainInput,
                             const std::vector<std::vector<double>>& trainOutput*/)
{
    //! @note Behövs inte pga finns i train?: if (trainingSetCount() == 0U) { return 0.0; }
    double sum{};

    for (std::size_t i{}; i < myTrainSetCount; ++i)
    {
        sum += averageError(myTrainInput[i], myTrainOutput[i]);
    }
    return 1.0 - sum / myTrainSetCount;
}

//! @todo fixa för accuracy()  -----------------------------------------------------------------------------
//! @todo implementera i headerfilerna.
double SingleLayer::averageError(const std::vector<double>& input,
                                 const std::vector<double>& reference)
{
    double error{};
    //! @todo Behöver predict() vara SingleLayer::predict()?
    const auto prediction{predict(input)};
    //! @todo Behövs ej?
    // checkVectorsMatching(reference, prediction);

    for (std::size_t i{}; i < prediction.size(); ++i)
    {
        // Tog absVal från project-1.
        error += absVal(reference[i] - prediction[i]);
    }
    return error / reference.size();
}


} // namespace ml::neural_network