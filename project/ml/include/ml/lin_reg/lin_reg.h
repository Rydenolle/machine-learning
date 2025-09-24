/**
 * @brief Linear regression implementation.
 */
#pragma once

#include "container/vector.h"
#include "ml/lin_reg/interface.h"
 
namespace ml
{
namespace lin_reg
{
/**
 * @brief Implement LinReg class by inheriting Interface.
 * 
 * @note LinReg class should not be inheritable.
 */
class LinReg final : public Interface
{
public:
    /**
     * @brief Create new LinReg.
     * 
     * @param[in] trainInput Reference to readable-only vector containing floats (input data for training).
     * @param[in] trainOutput Reference to readable-only vector containing floats (output data for training).
     */
    explicit LinReg(const container::Vector<double>& trainInput, const container::Vector<double>& trainOutput) noexcept;
    
    /**
     * @brief Delete the default constructor.
     */
    ~LinReg() noexcept override = default;

    /**
     * @brief Method for implementing a prediction with the model. 
     * 
     * @param[in] input The input data for basing the prediction upon.
     * 
     * @return The predicted value as a float. 
     */
    double predict(const double input) const noexcept override;

    /**
     * @brief Method for training the model.
     * 
     * @param[in] epochCount An unsigned integer indicating the amount of training epochs.
     * @param[in] learningRate Learning speed as a float, default 0.01 (1% learning speed).
     * 
     * @return Boolean true if training was implemented, false otherwise.
     */
    bool train(const size_t epochCount, const double learningRate = 0.01) noexcept;

    /**
     * @brief Delete the default constructor, delete copy and move constructors, delete operators.
     */
    LinReg()                            = delete;
    LinReg(const LinReg&)               = delete;
    LinReg(LinReg&&)                    = delete;
    LinReg& operator=(const LinReg&)    = delete;
    LinReg& operator=(LinReg&&)         = delete;

private:

    /** The training input data. */
    const container::Vector<double>& myTrainInput;

    /** The training output data. */
    const container::Vector<double>& myTrainOutput;

    /** The amount of complete training sets, where there's the same amount of input as output data. */
    const size_t myTrainSetCount;

    /** Bias value for the model (the 'm' in y = kx + m). */
    double myBias;
    
    /** Weight value for the model (the 'k' in y = kx + m). */
    double myWeight; 
};
} // namespace lin_reg
} // namespace ml