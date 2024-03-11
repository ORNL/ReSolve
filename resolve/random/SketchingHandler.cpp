// this is a virtual class
#include <resolve/vector/Vector.hpp>
#include <resolve/random/RandomSketchingCount.hpp>
#include <resolve/random/RandomSketchingCountCpu.hpp>
#include <resolve/random/RandomSketchingFWHT.hpp>
#include <resolve/random/RandomSketchingFWHTCpu.hpp>
#include "SketchingHandler.hpp"

namespace ReSolve {
  
  /**
   * @brief Constructor creates requested sketching method.
   *
   */ 
  SketchingHandler::SketchingHandler(SketchingMethod method, memory::MemorySpace memspace)
  {
    switch (method)
    {
      case LinSolverIterativeRandFGMRES::cs:
        if (memspace == memory::DEVICE) {
#ifdef RESOLVE_USE_GPU
          sketching_ = new RandomSketchingCount();
#endif
        } else {
          sketching_ = new RandomSketchingCountCpu();
        }
        break;
      
      case LinSolverIterativeRandFGMRES::fwht:
        if (memspace == memory::DEVICE) {
#ifdef RESOLVE_USE_GPU
          sketching_ = new RandomSketchingFWHT();
#endif
        } else {
          sketching_ = new RandomSketchingFWHTCpu();
        }
        break;
      
      default:
        break;
    }
  }

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
