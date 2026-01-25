/** @note Snyggt, Olle! */
/**
 * @brief Single Layer implementation details.
 */
#include <vector>

#include "ml/neural_network/single_layer.h"

namespace ml::neural_network
{
namespace
{
/** @note Snyggt att du lade en kommentar till denna. Jag kan dock säga att som regel är det OK/
 *        accepterat att inte kommentera "privata" funktioner i cpp-filerna (dessa är ej del av
 *        det "publika interfacet"), men det skadar ej.
 */
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
} // namespace

//--------------------------------------------------------------------------------
SingleLayer::SingleLayer(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                         const std::vector<std::vector<double>>& trainInput,
                         const std::vector<std::vector<double>>& trainOutput) noexcept
    /** @note Använd helst den universella initieraren {} i stället för () vid initiering.
     *        Parenteser är dock "the original way" (innan C++11) och fungerar givetvis fortfarande.
     */
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
    /** @note Lägg helst till parenteser runt dina villkor, dels för att tydliggöra avsikt,
     *        dels för att inte förlita sig på "operator precedence" (kommer du ihåg mattedjävulen?)
     *        samt att du slipper varningar från lint-verktyg. :)
     */
    if (0U == myTrainSetCount || 0U == epochCount || 0.0 >= learningRate) { return false; }

    // Train for 'epochCount' number of times.
    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        /** @note Initiera gärna i med {} i stället för =, men OK. :) */
        for (std::size_t i = 0U; i < myTrainSetCount; ++i)
        {
            // Get the values of the current training set.
            /** @note Här kommer jag inte kommentera på att du initierar med =, då auto och {}
             *        var struligt innan C++17 (det man skapade blev en "initializer list"). Nu
             *        är detta löst, men av denna anledning finns många programmerare som inte
             *        vill kombinera dessa, utan kör antingen int x{3} eller auto x = 3. :)
             */
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