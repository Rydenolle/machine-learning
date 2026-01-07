/**
 * @brief Dense layer stub for testing.
 */
#pragma once

#include <vector>

#include "ml/dense_layer/interface.h"

namespace ml::dense_layer
{
/**
 * @brief Dense layer stub for testing.
 */
class Stub final : public Interface
{
public:

    /**
     * @brief Create new dense layer.
     * 
     * @param[in] nodeCount The number of nodes in the dense layer.
     * @param[in] weightCount The number of weights per node in the dense layer.
     */
    explicit Stub(const std::size_t nodeCount, const std::size_t weightCount) noexcept;

    /**
     * @brief Delete the dense layer.
     */
    ~Stub() noexcept override = default;

    /**
     * @brief Get the number of nodes in the dense layer.
     * 
     * @return The number of nodes in the dense layer.
     */
    std::size_t nodeCount() const noexcept override;

    /**
     * @brief Get the number of weights per node in the dense layer.
     * 
     * @return The number of weights per node in the dense layer.
     */
    std::size_t weightCount() const noexcept override;

    /**
     * @brief Get the output values of the dense layer.
     * 
     * @return Vector holding the output values of the dense layer.
     */
    const std::vector<double>& output() const noexcept override;

    /**
     * @brief Get the error values of the dense layer.
     * 
     * @return Vector holding the error values of the dense layer.
     */
    const std::vector<double>& error() const noexcept override;

    /**
     * @brief Get the bias values of the dense layer.
     * 
     * @return Vector holding the bias values of the dense layer.
     */
    const std::vector<double>& bias() const noexcept override;

    /**
     * @brief Get the weights of the dense layer.
     * 
     * @return Vector holding the weights of the dense layer.
     */
    const std::vector<std::vector<double>>& weights() const noexcept override;

    /**
     * @brief Perform feedforward with the given input.
     * 
     * @param[in] input Input values with which to perform feedforward.
     * 
     * @return True if feedforward was performed, or false on error.
     */
    bool feedforward(const std::vector<double>& input) noexcept override;

    /**
     * @brief Perform backpropagation with the given reference values.
     * 
     *        This method is appropriate for output layers only.
     * 
     * @param[in] reference Reference values with which to perform backpropagation.
     * 
     * @return True if backpropagation was performed, or false on error.
     */
    bool backpropagate(const std::vector<double>& reference) noexcept override;

    /**
     * @brief Perform backpropagation with the given next layer.
     * 
     *        This method is appropriate for hidden layers only.
     * 
     * @param[in] nextLayer The next consecutive layer.
     * 
     * @return True if backpropagation was performed, or false on error.
     */
    bool backpropagate(const Interface& nextLayer) noexcept override;

    /**
     * @brief Perform optimization with the given input.
     * 
     * @param[in] input Input values with which to perform optimization.
     * @param[in] learningRate Learning rate to use for optimization.
     * 
     * @return True if optimization was performed, or false on error.
     */
    bool optimize(const std::vector<double>& input, const double learningRate) noexcept override;

    Stub()                       = delete; // No default constructor.
    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.

private:
    /** Stub values for output, error, and bias values. */
    std::vector<double> myNodes;

    /** Stub weights. */
    std::vector<std::vector<double>> myWeights;
};
} // namespace ml::dense_layer