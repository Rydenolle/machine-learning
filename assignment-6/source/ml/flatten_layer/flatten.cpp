/**
 * @brief Flatten layer implementation details.
 */
#include <vector>
#include <cstdlib>
#include <sstream>

#include "ml/flatten_layer/flatten.h"
#include "ml/types.h"
#include "ml/utils.h"
#include "ml/act_func/relu.h"
namespace ml::flatten_layer
{
//--------------------------------------------------------------------------------
FlattenLayer::FlattenLayer(const std::size_t inputSize)
    : myInputGradients{}
    , myOutput{}
    , myActFunc{}
{
    // Check the input size, throw if invalid.
    if (0U == inputSize)
    {
        throw std::invalid_argument("Cannot create flatten layer: invalid input size!");
    }

    // Initialize the matrices - set output size to input size ^ 2.
    initMatrix(myInputGradients, inputSize);
    initMatrix(myOutput, inputSize * inputSize);
}

//--------------------------------------------------------------------------------
std::size_t FlattenLayer::inputSize() const noexcept { return myInputGradients.size(); }

//--------------------------------------------------------------------------------
std::size_t FlattenLayer::outputSize() const noexcept { return myOutput.size(); }

//--------------------------------------------------------------------------------
const Matrix2d& FlattenLayer::inputGradients() const noexcept { return myInputGradients; }

//--------------------------------------------------------------------------------
const Matrix1d& FlattenLayer::output() const noexcept { return myOutput; }

//--------------------------------------------------------------------------------
bool FlattenLayer::feedforward(const Matrix2d& input) noexcept
{
    // Get the input size.
    const std::size_t inputSize{myInputGradients.size()};

    // Check the input matrix, return false on dimension mismatch.
    if ((input.size() != inputSize) || !isMatrixSquare(input)) { return false; }

    // Flatten the input: [i][j] => [inputSize * i + j].
    for (std::size_t i{}; i < inputSize; ++i)
    {
        for (std::size_t j{}; j < inputSize; ++j)
        {
            myOutput[inputSize * i + j] = input[i][j];
        }
    }
    // Return true to indicate success.
    return true;
}

//--------------------------------------------------------------------------------
bool FlattenLayer::backpropagate(const Matrix1d& outputGradients) noexcept
{
    // Check the output matrix, return false on dimension mismatch.
    if (outputGradients.size() != myOutput.size()) { return false; }

    // Get the input size.
    const std::size_t inputSize{myInputGradients.size()};

    // Unflatten the input: [inputSize * i + j] => [i][j].
    for (std::size_t i{}; i < inputSize; ++i)
    {
        for (std::size_t j{}; j < inputSize; ++j)
        {
            myInputGradients[i][j] = outputGradients[inputSize * i + j];
        }
    }
    // Return true to indicate success.
    return true;
}

} // namespace ml::flatten_layer
