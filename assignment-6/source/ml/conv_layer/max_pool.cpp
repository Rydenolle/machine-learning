/**
 * @brief Max pooling layer implementation details.
 */

 //! @todo Add the implementation details of the ml::conv_layer::MaxPool class here.
#include <vector>
#include <cstdlib>
#include <sstream>

#include "ml/conv_layer/max_pool.h"
#include "ml/types.h"
#include "ml/utils.h"
#include "ml/act_func/relu.h"

namespace ml::conv_layer
{
class MaxPoolLayerLayer final : public Interface
{
public:
    //--------------------------------------------------------------------------------
    std::size_t inputSize() const noexcept override;

    //--------------------------------------------------------------------------------
    std::size_t outputSize() const noexcept override;

    //--------------------------------------------------------------------------------
    const Matrix2d& output() const noexcept override;

    //--------------------------------------------------------------------------------
    const Matrix2d& inputGradients() const noexcept override;

    //--------------------------------------------------------------------------------
    bool feedforward(const Matrix2d& input) noexcept override;

    //--------------------------------------------------------------------------------
    bool backpropagate(const Matrix2d& outputGradients) noexcept override;

    //--------------------------------------------------------------------------------
    bool optimize(double learningRate) noexcept override;


private:
//! @todo inget här
} // namespace ml::conv_layer