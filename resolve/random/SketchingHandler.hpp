/**
 * @file SketchingHandler.hpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of SketchingHandler class
 *
 */
#pragma once
#include <resolve/LinSolverIterativeRandFGMRES.hpp>

namespace ReSolve
{
  // Forward declarations
  class RandomSketchingImpl;

  namespace vector
  {
    class VectorHandler;
  }

  /**
   * @brief Class that invokes sketching method using PIMPL idiom.
   *
   */
  class SketchingHandler
  {
  private:
    using SketchingMethod = LinSolverIterativeRandFGMRES::SketchingMethod;
    using vector_type     = vector::Vector;

  public:
    SketchingHandler(SketchingMethod method, memory::DeviceType devtype);
    ~SketchingHandler();

    /// Actual sketching process
    int Theta(vector_type* input, vector_type* output);

    /// Setup the parameters, sampling matrices, permuations, etc.
    int setup(index_type n, index_type k);

    /// Needed for iterative methods with restarting
    int reset();

  private:
    RandomSketchingImpl* sketching_{nullptr}; ///< Pointer to implementation
  };

} // namespace ReSolve
