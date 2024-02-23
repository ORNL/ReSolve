// this is a virtual class
#include <resolve/vector/Vector.hpp>
#include "RandSketchingManager.hpp"

namespace ReSolve {
  
  // constructor
  /**
   * @brief Simple constructor
   *
   */ 
   RandSketchingManager::RandSketchingManager()
   {
     k_rand_ = 0;
     n_ = 0;
     N_ = 0;
   }

  // destructor
  /**
   * @brief Destructor
   *
   */ 
  RandSketchingManager::~RandSketchingManager()
  {
  }

}
