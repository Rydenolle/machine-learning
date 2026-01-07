/**
 * @brief Dense layer stub implementation for testing.
 */
#include <tuple>
#include <vector>

#include "ml/dense_layer/stub.h"

namespace ml::dense_layer
{
// -----------------------------------------------------------------------------
Stub::Stub(const std::size_t nodeCount, const std::size_t weightCount) noexcept
    : myNodes(nodeCount, 0.5)
    , myWeights(nodeCount, std::vector<double>(nodeCount, 0.5))
{}

// -----------------------------------------------------------------------------
std::size_t Stub::nodeCount() const noexcept { return myNodes.size(); }

// -----------------------------------------------------------------------------
std::size_t Stub::weightCount() const noexcept 
{ 
    return myWeights.empty() ? 0U : myWeights[0U].size(); 
}

// -----------------------------------------------------------------------------
const std::vector<double>& Stub::output() const noexcept { return myNodes; }

// -----------------------------------------------------------------------------
const std::vector<double>& Stub::error() const noexcept { return myNodes; }

// -----------------------------------------------------------------------------
const std::vector<double>& Stub::bias() const noexcept { return myNodes; }

// -----------------------------------------------------------------------------
const std::vector<std::vector<double>>& Stub::weights() const noexcept { return myWeights; }

// -----------------------------------------------------------------------------
bool Stub::feedforward(const std::vector<double>& input) noexcept 
{ 
    std::ignore = input;
    return true; 
}

// -----------------------------------------------------------------------------
bool Stub::backpropagate(const std::vector<double>& reference) noexcept
{
    std::ignore = reference;
    return true; 
}

// -----------------------------------------------------------------------------
bool Stub::backpropagate(const Interface& nextLayer) noexcept
{
    std::ignore = nextLayer;
    return true;
}

// -----------------------------------------------------------------------------
bool Stub::optimize(const std::vector<double>& input, const double learningRate) noexcept 
{
    std::ignore = input;
    std::ignore = learningRate;
    return true;
}
} // namespace ml::dense_layer