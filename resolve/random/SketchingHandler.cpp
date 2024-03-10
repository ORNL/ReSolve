// this is a virtual class
#include <resolve/vector/Vector.hpp>
#include <resolve/random/RandSketchingCountSketch.hpp>
#include <resolve/random/RandSketchingFWHT.hpp>
#include "SketchingHandler.hpp"

namespace ReSolve {
  
  // constructor
  /**
   * @brief Simple constructor
   *
   */ 
  SketchingHandler::SketchingHandler(SketchingMethod method, memory::MemorySpace memspace)
  {
    // if (vh.getIsCudaEnabled()) {
    // }
    switch (method)
    {
      case LinSolverIterativeRandFGMRES::cs:
        sketching_ = new RandSketchingCountSketch(memspace);
        /* code */
        break;
      
      case LinSolverIterativeRandFGMRES::fwht:
        sketching_ = new RandSketchingFWHT(memspace);
        /* code */
        break;
      
      default:
        break;
    }
  }

  // destructor
  /**
   * @brief Destructor
   *
   */ 
  SketchingHandler::~SketchingHandler()
  {
    delete sketching_;
  }

  int SketchingHandler::Theta(vector_type* input, vector_type* output)
  {
    return sketching_->Theta(input, output);
  }

  int SketchingHandler::setup(index_type n, index_type k)
  {
    return sketching_->setup(n, k);
  }

  int SketchingHandler::reset()
  {
    return sketching_->reset();
  }

}
