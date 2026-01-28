/**
 * @brief Max pooling layer implementation.
 */
#pragma once

#include <cstdlib>
#include <vector>

#include "ml/conv_layer/interface.h"
#include "ml/types.h"
#include "ml/utils.h"
#include "ml/act_func/relu.h"


namespace ml::conv_layer
{
class MaxPoolLayerLayer final : public Interface
{
public:
    /**
     * @brief Destructor. '
     */
    ~MaxPoolLayer() noexcept override = default;

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
    MaxPoolLayer()                           = delete;
    MaxPoolLayer(const MaxPoolLayer&)             = delete;
    MaxPoolLayer(MaxPoolLayer&&)                  = delete;
    MaxPoolLayer& operator=(const MaxPoolLayer&)  = delete;
    MaxPoolLayer& operator=(MaxPoolLayer&&)       = delete;

private:
    /** Input matrix. */
    Matrix2d myInput;

    /** Input gradient matrix. */
    Matrix2d myInputGradients;

    /** Output matrix. */
    Matrix2d myOutput;
};
} // namespace ml::conv_layer

//! @todo Implement the ml::conv_layer::MaxPoolLayer class here.

//! @todo When you have implemented this class, update the factory (source/ml/factory/factory.cpp) 
//!       to use it instead of the stub.
