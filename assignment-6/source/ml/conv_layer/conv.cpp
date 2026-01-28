/**
 * @brief Convolutional layer implementation details.
 */
#include <cstdlib>
#include <sstream>

#include "ml/conv_layer/conv.h"
#include "ml/types.h"
#include "ml/utils.h"
#include "ml/act_func/relu.h"

namespace ml::conv_layer
{
//--------------------------------------------------------------------------------
ConvLayer::ConvLayer(const std::size_t inputSize, const std::size_t kernelSize)
    : myInputPadded{}
    , myInputGradientsPadded{}
    , myInputGradients{}
    , myKernel{}
    , myKernelGradients{}
    , myOutput{}
    , myBias{randomStartVal()}
    , myBiasGradient{}
    , myActFunc{}
{
    // Implement kernel min and max size. Min size can't be 0.
    constexpr std::size_t minKernelSize{1U};
    constexpr std::size_t maxKernelSize{11U};

    // Throw exception if the kernel size is outside range [1, 11] or larger than the input size.
    if ((minKernelSize > kernelSize) || (maxKernelSize < kernelSize))
    {
        std::stringstream msg{};
        msg << "Invalid kernel size " << kernelSize << ": kernel size must be in range ["
            << minKernelSize << ", " << maxKernelSize << "]!\n";
        throw std::invalid_argument(msg.str());
    }
    else if (inputSize < kernelSize)
    {
        throw std::invalid_argument(
            "Failed to create convolutional layer: kernel size cannot be greater than input size!");
    }

    // Calculate pad offset and padded size.
    const std::size_t padOffset{kernelSize / 2U};
    const std::size_t paddedSize{inputSize + 2U * padOffset};

    // Initialize the matrices with zeros.
    initMatrix(myInputPadded, paddedSize);
    initMatrix(myInputGradients, inputSize);
    initMatrix(myInputGradientsPadded, paddedSize);
    initMatrix(myKernel, kernelSize);
    initMatrix(myKernelGradients, kernelSize);
    initMatrix(myOutput, inputSize);
}

//--------------------------------------------------------------------------------
std::size_t ConvLayer::inputSize() const noexcept { return myInputGradients.size(); }

//--------------------------------------------------------------------------------
std::size_t ConvLayer::outputSize() const noexcept { return myOutput.size(); }

//--------------------------------------------------------------------------------
const Matrix2d& ConvLayer::output() const noexcept { return myOutput; }

//--------------------------------------------------------------------------------
const Matrix2d& ConvLayer::inputGradients() const noexcept { return myInputGradients; }

//--------------------------------------------------------------------------------
bool ConvLayer::feedforward(const Matrix2d& input) noexcept
{
    // Check the input matrix, return false on dimension mismatch.
    if ((input.size() != myOutput.size()) || !isMatrixSquare(input)) { return false; }

    // Pad the input with zeros.
    padInput(input);

    // Run feedforward; accumulate bias and contributions from the input and the kernel.
    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            // Start by adding the bias value.
            auto sum{myBias};

            // Iterate through the kernel and add the input * kernel values.
            for (std::size_t ki{}; ki < myKernel.size(); ++ki)
            {
                for (std::size_t kj{}; kj < myKernel.size(); ++kj)
                {
                    sum += myInputPadded[i + ki][j + kj] * myKernel[ki][kj];
                }
            }

            // Pass the sum through the ReLU activation function, store as output.
            myOutput[i][j] = myActFunc.output(sum);
        }
    }
    return true;
}

//--------------------------------------------------------------------------------
bool ConvLayer::backpropagate(const Matrix2d& outputGradients) noexcept
{
    // Check the output gradients matrix, return false on dimension mismatch.
    if ((outputGradients.size() != myOutput.size()) || !isMatrixSquare(outputGradients))
    {
        return false;
    }

    // Reinitialize the gradients with zeros (to remove old values).
    // Else values from the previous backpropagation would still remain.
    initMatrix(myInputGradientsPadded);
    initMatrix(myInputGradients);
    initMatrix(myKernelGradients);
    myBiasGradient = 0.0;

    // Iterate through the output gradients.
    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            // Calculate output derivate.
            const auto delta{outputGradients[i][j] * myActFunc.delta(myOutput[i][j])};

            // Accumulate the bias gradient by adding all output delta values.
            myBiasGradient += delta;

            // Iterate through the kernel.
            for (std::size_t ki{}; ki < myKernel.size(); ++ki)
            {
                for (std::size_t kj{}; kj < myKernel.size(); ++kj)
                {
                    myKernelGradients[ki][kj] += myInputPadded[i + ki][j + kj] * delta;
                    myInputGradientsPadded[i + ki][j + kj] += myKernel[ki][kj] * delta;
                }
            }
        }
    }
    // Extract input gradients without zeros.
    extractInputGradients();
    return true;
}

//--------------------------------------------------------------------------------
bool ConvLayer::optimize(double learningRate) noexcept
{
    // Check the learning rate, return false if out of range.
    if ((0.0 >= learningRate) || (1.0 < learningRate)) { return false; }

    // Adjust the bias with the computed bias gradient, multiplied by the learning rate.
    // We subtract, since the gradients are computed in this manner, as opposed to what
    // we've used in dense layer.
    myBias -= myBiasGradient * learningRate;

    // Adjust the kernel weights with the corresponding gradients and the learning rate.
    for (std::size_t ki{}; ki < myKernel.size(); ++ki)
    {
        for (std::size_t kj{}; kj < myKernel.size(); ++kj)
        {
            myKernel[ki][kj] -= myKernelGradients[ki][kj] * learningRate;
        }
    }
    return true;
}

} // namespace ml::conv_layer
