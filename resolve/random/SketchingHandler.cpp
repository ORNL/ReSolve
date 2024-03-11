// this is a virtual class
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
   */ 
  SketchingHandler::SketchingHandler(SketchingMethod method, memory::DeviceType devtype)
  {
    switch (method) {
      case LinSolverIterativeRandFGMRES::cs:
        switch(devtype) {
          case memory::CUDADEVICE:
#ifdef RESOLVE_USE_CUDA
            sketching_ = new RandomSketchingCountCuda();
#endif
            break;
          case memory::HIPDEVICE:
#ifdef RESOLVE_USE_HIP
            sketching_ = new RandomSketchingCountHip();
#endif
            break;
          case memory::NONE:
            sketching_ = new RandomSketchingCountCpu();
            break;
          default:
            sketching_ = nullptr;
            break;
        }
        break;    

      case LinSolverIterativeRandFGMRES::fwht:
        switch(devtype) {
          case memory::CUDADEVICE:
#ifdef RESOLVE_USE_CUDA
            sketching_ = new RandomSketchingFWHTCuda();
#endif
            break;
          case memory::HIPDEVICE:
#ifdef RESOLVE_USE_HIP
            sketching_ = new RandomSketchingFWHTHip();
#endif
            break;
          case memory::NONE:
            sketching_ = new RandomSketchingFWHTCpu();
            break;
          default:
            sketching_ = nullptr;
            break;
        }
        break;      

      default:
        sketching_ = nullptr;
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
