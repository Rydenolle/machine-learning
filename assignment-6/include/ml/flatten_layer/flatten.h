/**
 * @brief Flatten layer implementation.
 */
#pragma once

#include <cstdlib>
#include <vector>

#include "ml/flatten_layer/interface.h"
#include "ml/act_func/relu.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::flatten_layer
{
/**
 * @brief Flatten layer interface.
 */
class FlattenLayer final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     */
    explicit FlattenLayer(const std::size_t inputSize);

    /** 
     * @brief Destructor. 
     */
    ~FlattenLayer() noexcept override = default;

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
     * @brief Get the input gradients of the layer.
     * 
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override;

    /**
     * @brief Get the flattened output of the layer.
     * 
     * @return Matrix holding the output of the layer.
     */
    const Matrix1d& output() const noexcept override;

    /**
     * @brief Flatten the input from 2D to 1D.
     * 
     * @param[in] input Matrix holding input data.
     * 
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override;

     /**
     * @brief Unflatten the output gradients from 1D to 2D.
     * 
     * @param[in] outputGradients Matrix holding output gradients.
     * 
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& outputGradients) noexcept override;

    /**
     * @brief Delete the default constructor, delete copy and move constructors, delete operators.
     */
    FlattenLayer()                                  = delete;
    FlattenLayer(const FlattenLayer&)               = delete;
    FlattenLayer(FlattenLayer&&)                    = delete;
    FlattenLayer& operator=(const FlattenLayer&)    = delete;
    FlattenLayer& operator=(FlattenLayer&&)         = delete;

private:

        
    /** Unflattened input gradients (to pass to the previous layer). */
    Matrix2d myInputGradients;

    /** Flattened output (to pass to the next layer). */
    Matrix1d myOutput;

    /** Relu. */
    act_func::Relu myActFunc;
};
} // namespace ml::flatten_layer
