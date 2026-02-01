/**
 * @brief Max pooling layer implementation details.
 */
#include <vector>
#include <cstdlib>
#include <sstream>

#include "ml/conv_layer/max_pool.h"
#include "ml/types.h"
#include "ml/utils.h"
#include "ml/act_func/relu.h"

namespace ml::conv_layer
{

MaxPoolLayer::MaxPoolLayer(const std::size_t inputSize, const std::size_t poolSize)
    : myInput{}
    , myInputGradients{}
    , myOutput{}
    , myActFunc{}
{
     // Check the input arguments, throw an exception if invalid.
    if ((0U == inputSize) || (0U == poolSize) || (0U != (inputSize % poolSize)))
    {
        throw std::invalid_argument(
            "Cannot create max pooling layer: invalid input arguments!");
    }

    // Compute the output size.
    const std::size_t outputSize{inputSize / poolSize};

    // Initialize the matrices.
    initMatrix(myInput, inputSize);
    initMatrix(myInputGradients, inputSize);
    initMatrix(myOutput, outputSize);
}

//--------------------------------------------------------------------------------
std::size_t MaxPoolLayer::inputSize() const noexcept { return myInputGradients.size(); }

//--------------------------------------------------------------------------------
std::size_t MaxPoolLayer::outputSize() const noexcept { return myOutput.size(); }

//--------------------------------------------------------------------------------
const Matrix2d& MaxPoolLayer::output() const noexcept { return myOutput; }

//--------------------------------------------------------------------------------
const Matrix2d& MaxPoolLayer::inputGradients() const noexcept { return myInputGradients; }

//--------------------------------------------------------------------------------
bool MaxPoolLayer::feedforward(const Matrix2d& input) noexcept
{
    // Check the input matrix, return false on dimension mismatch.
    if ((input.size() != myInput.size()) || !isMatrixSquare(input)) { return false; }

    // Calculate the pool size.
    const std::size_t poolSize{input.size() / myOutput.size()};

    // Iterate through the image pool by pool, find and store the max value.
    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            // Get the input row and column.
            const std::size_t inRow{i * poolSize};
            const std::size_t inCol{j * poolSize};

            // Use the first value as max value, compare with the other values in the pool.
            double maxVal{input[inRow][inCol]};

            // Iterate through the pool.
            for (std::size_t pi{}; pi < poolSize; ++pi)
            {
                for (std::size_t pj{}; pj < poolSize; ++pj)
                {
                    // Get the value at the current cell.
                    const auto val{input[inRow + pi][inCol + pj]};

                    // Compare the value with the local max, store the bigger one.
                    if (val > maxVal) { maxVal = val; }
                }
            }
            // Store the max value in the output matrix.
            myOutput[i][j] = maxVal;
        }
    }
    // Store the input for backpropagation.
    myInput = input;

    // Return true to indicate success.
    return true;
}

//--------------------------------------------------------------------------------
bool MaxPoolLayer::backpropagate(const Matrix2d& outputGradients) noexcept
{
    // Check the output gradient matrix, return false on dimension mismatch.
    if ((outputGradients.size() != myOutput.size()) || !isMatrixSquare(outputGradients))
    {
        return false;
    }

    // Calculate the pool size.
    const std::size_t poolSize{myInput.size() / myOutput.size()};

    // Reinitialize input matrix with zeros (remove leftovers from previous backpropagation).
    initMatrix(myInputGradients);

    // Locate the max value coordinates (row, col) and place the gradients there.
    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            // Compute the input row and column.
            const std::size_t inRow{i * poolSize};
            const std::size_t inCol{j * poolSize};

            // Get the max value for comparison.
            const auto maxVal{myOutput[i][j]};

            // Variables holding the max coordinates (start with the first call of the pool).
            std::size_t maxRow{inRow};
            std::size_t maxCol{inCol};

            // Indicate whether the max value has been found.
            bool found{false};

            for (std::size_t pi{}; pi < poolSize; ++pi)
            {
                for (std::size_t pj{}; pj < poolSize; ++pj)
                {
                    // Get the value of the current cell.
                    const auto val{myInput[inRow + pi][inCol + pj]};

                    // If this is the max value, store the coordinates.
                    if (val == maxVal)
                    {
                        // Store the coordinates of the max value.
                        maxRow = inRow + pi;
                        maxCol = inCol + pj;

                        // Indicate that the value has been found and break the 'pj' loop.
                        found = true;
                        break;
                    }
                }
                // Break the 'pi' loop if the max value has been found.
                if (found) { break; }
            }
            // Write the output gradient to the max value position.
            myInputGradients[maxRow][maxCol] = outputGradients[i][j];
        }
    }
    // Return true to indicate success.
    return true;
}

//--------------------------------------------------------------------------------
bool MaxPoolLayer::optimize(double learningRate) noexcept { return true; }

} // namespace ml::conv_layer