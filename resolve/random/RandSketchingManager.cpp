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

  /**
   * @brief Returns the size of base (non-sketched) vector.
   *
   * @return Base vector size _n_.
   */ 
  index_type RandSketchingManager::getVectorSize()
  {
    return n_;
  }
  
  /**
   * @brief Returns the size of sketched vector.
   *
   * @return Sketched vector size _k_, generally _k_ =< _n_.
   */ 
  index_type RandSketchingManager::getSketchSize()
  {
    return k_rand_;
  }

  /**
   * @brief If padding is used, returns size of padded vector.
   *
   * @return Sketched vector size _N_, generally _N_ >= _n_.
   */ 
  index_type RandSketchingManager::getPaddedSize()
  {
    return N_;
  }
}
