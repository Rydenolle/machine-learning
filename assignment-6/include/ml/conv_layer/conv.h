/**
 * @brief Convolutional layer implementation.
 */
#pragma once

#include <cstdlib>
#include <vector>

#include "ml/act_func/relu.h"
#include "ml/conv_layer/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::conv_layer
{
class ConvLayer final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size as a size_t. Must be > 0.
     * @param[in] kernelSize Kernel size as a size_t. Must be > 0 and < input size
     */
    explicit ConvLayer(const std::size_t inputSize, const std::size_t kernelSize);

    /**
     * @brief Destructor.
     */
    ~ConvLayer() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     * 
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override;

    /**
     * @brief Get the output size of the layer.
     * 
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override;

    /**
     * @brief Get the output of the layer.
     * 
     * @return Matrix holding the output of the layer.
     */
    const Matrix2d& output() const noexcept override;

    /**
     * @brief Get the input gradients of the layer.
     * 
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override;

    /**
     * @brief Perform feedforward operation.
     * 
     * @param[in] input Matrix holding input data.
     * 
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override;

    /**
     * @brief Perform backpropagation.
     * 
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     * 
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix2d& outputGradients) noexcept override;

    /**
     * @brief Perform optimization.
     * 
     * @param[in] learningRate Learning rate to use.
     * 
     * @return True on success, false on failure.
     */
    bool optimize(double learningRate) noexcept override;

    /**
     * @brief Delete the default constructor, delete copy and move constructors, delete operators.
     */
    ConvLayer()                             = delete;
    ConvLayer(const ConvLayer&)             = delete;
    ConvLayer(ConvLayer&&)                  = delete;
    ConvLayer& operator=(const ConvLayer&)  = delete;
    ConvLayer& operator=(ConvLayer&&)       = delete;

private:
    /**
     * @brief Pad input with zeros.
     *
     * @param[in] input Input data.
     */
    void padInput(const Matrix2d& input) noexcept
    {
        // Compute the pad offset (the number of zeros in each direction).
        const std::size_t padOffset{myKernel.size() / 2U};

        // Ensure that the padded input matrix is filled with zeros only.
        initMatrix(myInputPadded);

        // Copy the input values to the corresponding padded matrix.
        for (std::size_t i{}; i < myOutput.size(); ++i)
        {
            for (std::size_t j{}; j < myOutput.size(); ++j)
            {
                myInputPadded[i + padOffset][j + padOffset] = input[i][j];
            }
        }
    }

    /**
     * @brief Extract input gradients.
     */
    void extractInputGradients() noexcept
    {
        // Compute the pad offset (the number of zeros in each direction).
        const std::size_t padOffset{myKernel.size() / 2U};

        for (std::size_t i{}; i < myOutput.size(); ++i)
        {
            for (std::size_t j{}; j < myOutput.size(); ++j)
            {
                myInputGradients[i][j] = myInputGradientsPadded[i + padOffset][j + padOffset];
            }
        }
    }

    /** Input matrix (padded with zeros). */
    Matrix2d myInputPadded;

    /** Input gradient matrix (padded with zeros). */
    Matrix2d myInputGradientsPadded;

    /** Input gradient matrix (without padding). */
    Matrix2d myInputGradients;

    /** Kernel matrix (holding weights). */
    Matrix2d myKernel;

    /** Kernel gradient matrix. */
    Matrix2d myKernelGradients;

    /** Output matrix. */
    Matrix2d myOutput;

    /** Bias value. */
    double myBias;

    /** Bias gradient. */
    double myBiasGradient;

    /** Relu. */
    act_func::Relu myActFunc;

};
} // namespace ml
