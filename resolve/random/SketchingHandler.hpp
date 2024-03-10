#pragma once
#include <resolve/LinSolverIterativeRandFGMRES.hpp>

namespace ReSolve
{
  class RandSketchingManager;
  namespace vector
  {
    class VectorHandler;
  }

  class SketchingHandler
  {
    private:
      using SketchingMethod = LinSolverIterativeRandFGMRES::SketchingMethod;
      using vector_type = vector::Vector;
    public:
      SketchingHandler(SketchingMethod method, memory::MemorySpace memspace);
      ~SketchingHandler();

      // Actual sketching process
      int Theta(vector_type* input, vector_type* output);

      // Setup the parameters, sampling matrices, permuations, etc
      int setup(index_type n, index_type k);
      // Need to use with methods that restart
      int reset();

    private:
      RandSketchingManager* sketching_{nullptr}; ///< Pointer to implementation

      bool isCpuEnabled_{false};
      bool isCudaEnabled_{false};
      bool isHipEnabled_{false};
  };

} // namespace ReSolve