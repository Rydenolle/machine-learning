//! @note Snyggt! Det enda jag kan anmärka på är att du inte behöver använda aktiveringsfunktioner,
//!       då maxpool-lager inte innehåller träningsbara parametrar, se nedan.

/**
 * @brief Max pooling layer implementation.
 */
#pragma once

#include <cstdlib>
#include <vector>

//! @note ml/act_func/relu.h behöver inte inkluderas, då maxpool-lager inte innehåller
//!       träningsbara parametrar.
#include "ml/conv_layer/interface.h"
#include "ml/act_func/relu.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::conv_layer
{
class MaxPoolLayer final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     */
    explicit MaxPoolLayer(const std::size_t inputSize, const std::size_t poolSize);

    /**
     * @brief Destructor.
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
    MaxPoolLayer()                                = delete;
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

    //! @note Detta attribut bör tas bort!
    /** Relu. */
    act_func::Relu myActFunc;
};
} // namespace ml::conv_layer
