/**
 * @brief Single Layer implementation details.
 */
#include <vector>

#include "ml/neural_network/single_layer.h"

namespace ml::neural_network
{
namespace
{
//--------------------------------------------------------------------------------
std::size_t getTrainSetCount(const std::vector<std::vector<double>>& input,
                             const std::vector<std::vector<double>>& output) noexcept
{
    return (input.size() <= output.size() ? input.size() : output.size());
}

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
    return true;

}
} // namespace ml::neural_network