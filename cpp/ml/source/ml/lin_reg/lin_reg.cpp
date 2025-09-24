/**
 * @brief Linear regression implementation details.
 */
#include "container/vector.h"
#include "ml/lin_reg/lin_reg.h"

namespace ml
{
namespace lin_reg
{
namespace
{
//--------------------------------------------------------------------------------
size_t getTrainSetCount(const container::Vector<double>& input, 
                             const container::Vector<double>& output) noexcept
{
    return (input.size() <= output.size()) ? input.size() : output.size();
}

/** Default start value for the weight and bias. */
constexpr auto DefaultStartVal{0.5};

} // namespace

//--------------------------------------------------------------------------------
LinReg::LinReg(const container::Vector<double>& trainInput, const container::Vector<double>& trainOutput) noexcept
    : myTrainInput{trainInput}
    , myTrainOutput{trainOutput}
    , myTrainSetCount{getTrainSetCount(trainInput, trainOutput)}
    , myBias{DefaultStartVal}
    , myWeight{DefaultStartVal}
{} // :)

//--------------------------------------------------------------------------------
double LinReg::LinReg::predict(const double input) const noexcept
{
    return myWeight * input + myBias;
}

//--------------------------------------------------------------------------------
bool LinReg::LinReg::train(const size_t& epochCount, const double& learningRate) noexcept
{
    if ((0U == epochCount) || (0.0 <= learningRate)) { return false; }
    
    // Train the model here.
    return true;
}
} // namespace lin_reg
} // namespace ml