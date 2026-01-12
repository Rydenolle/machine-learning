/**
 * @brief Neural network interface implementation.
 */
#pragma once

#include <vector>

namespace ml::neural_network
{
/**
 * @brief Neural network interface implementation.
 */
class Interface
{
public:
    /**
     * @brief Delete the default destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Virtual method for implementing a prediction with the model.
     *
     * @param[in] input The input data for basing the prediction upon.
     *
     * @return The predicted value as a vector containing float numbers.
     */
    virtual const std::vector<double>& predict(const std::vector<double>& input) = 0;
};
} // namespace ml::neural_network
