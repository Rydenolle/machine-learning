/**
 * @brief Linear regression interface implementation. 
 */
#pragma once

namespace ml
{
namespace lin_reg
{
/**
 * @brief Linear regression interface implementation.
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
     * @return The predicted value as a float. 
     */
    virtual double predict(const double input) const = 0;
};
} // namespace lin_reg
} // namespace ml