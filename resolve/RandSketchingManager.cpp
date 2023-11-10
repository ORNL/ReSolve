// this is a virtual class
#include "RandSketchingManager.hpp"

namespace ReSolve {
  
  // constructor
   RandSketchingManager::RandSketchingManager()
   {
     k_rand_ = 0;
     n_ = 0;
     N_ = 0;
   }

  // destructor
  RandSketchingManager::~RandSketchingManager()
  {
  }

  index_type RandSketchingManager::getVectorSize()
  {
    return n_;
  }
  
  index_type RandSketchingManager::getSketchSize()
  {
    return k_rand_;
  }

  index_type RandSketchingManager::getPaddedSize()
  {
    return N_;
  }
}
