// this is a virtual class
#include <resolve/vector/Vector.hpp>
#include <resolve/random/RandomSketchingCount.hpp>
#include <resolve/random/RandomSketchingCountCpu.hpp>
#include <resolve/random/RandomSketchingFWHT.hpp>
#include <resolve/random/RandomSketchingFWHTCpu.hpp>
#include "SketchingHandler.hpp"

namespace ReSolve {
  
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
        if (memspace == memory::DEVICE) {
          sketching_ = new RandomSketchingCount();
        } else {
          sketching_ = new RandomSketchingCountCpu();
        }
        break;
      
      case LinSolverIterativeRandFGMRES::fwht:
        if (memspace == memory::DEVICE) {
          sketching_ = sketching_ = new RandomSketchingFWHT();
        } else {
          sketching_ = sketching_ = new RandomSketchingFWHTCpu();
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
