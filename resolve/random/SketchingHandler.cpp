/**
 * @file SketchingHandler.cpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Implementation of the SketchingHandler class
 * 
 */
#include <resolve/vector/Vector.hpp>
#include <resolve/random/RandomSketchingCountCuda.hpp>
#include <resolve/random/RandomSketchingCountHip.hpp>
#include <resolve/random/RandomSketchingCountCpu.hpp>
#include <resolve/random/RandomSketchingFWHTCuda.hpp>
#include <resolve/random/RandomSketchingFWHTHip.hpp>
#include <resolve/random/RandomSketchingFWHTCpu.hpp>
#include "SketchingHandler.hpp"

namespace ReSolve {
  
  /**
   * @brief Constructor creates requested sketching method.
   * 
   * Create instance of the specified sketching method on the selected device.
   *
   */ 
  SketchingHandler::SketchingHandler(SketchingMethod method, memory::DeviceType devtype)
  {
    if (devtype == memory::NONE) {
      switch (method) {
        case LinSolverIterativeRandFGMRES::cs:
          sketching_ = new RandomSketchingCountCpu();
          break;    
        case LinSolverIterativeRandFGMRES::fwht:
          sketching_ = new RandomSketchingFWHTCpu();
          break;
        default:
          sketching_ = nullptr;
          break;
      }
    }

#ifdef RESOLVE_USE_CUDA
    if (devtype == memory::CUDADEVICE) {
      switch (method) {
        case LinSolverIterativeRandFGMRES::cs:
          sketching_ = new RandomSketchingCountCuda();
          break;    
        case LinSolverIterativeRandFGMRES::fwht:
          sketching_ = new RandomSketchingFWHTCuda();
          break;
        default:
          sketching_ = nullptr;
          break;
      }
    }
#endif

#ifdef RESOLVE_USE_HIP
    if (devtype == memory::HIPDEVICE) {
      switch (method) {
        case LinSolverIterativeRandFGMRES::cs:
          sketching_ = new RandomSketchingCountHip();
          break;    
        case LinSolverIterativeRandFGMRES::fwht:
          sketching_ = new RandomSketchingFWHTHip();
          break;
        default:
          sketching_ = nullptr;
          break;
      }
    }
#endif

  }

  /**
   * @brief Destructor deletes the sketching method implementation.
   *
   */ 
  SketchingHandler::~SketchingHandler()
  {
    delete sketching_;
  }

  /// Calls sketching process.
  int SketchingHandler::Theta(vector_type* input, vector_type* output)
  {
    return sketching_->Theta(input, output);
  }

  /// Calls initial setup.
  int SketchingHandler::setup(index_type n, index_type k)
  {
    return sketching_->setup(n, k);
  }

  /// Resets sketching method (e.g. when the iterative method restarts).
  int SketchingHandler::reset()
  {
    return sketching_->reset();
  }

}
