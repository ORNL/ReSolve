/**
 * @file RandomSketchingCountCpu.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of RandomSketchingCountCuda class.
 * 
 */
#pragma once
#include <resolve/Common.hpp>
#include <resolve/random/RandomSketchingImpl.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve {

  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  /**
   * @brief Count sketching implementation for CPU.
   * 
   */
  class RandomSketchingCountCpu : public RandomSketchingImpl
  {
    private:
      using vector_type = vector::Vector;

    public: 
      // constructor
      RandomSketchingCountCpu();

      // destructor
      virtual ~RandomSketchingCountCpu();

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output);

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k);
      virtual int reset(); // if needed can be reset (like when Krylov method restarts)

    private:
      index_type n_{0};      ///< size of base vector
      index_type k_rand_{0}; ///< size of sketched vector

      index_type* h_labels_{nullptr}; ///< label array size _n_, with values from _0_ to _k-1_ assigned by random
      index_type* h_flip_{nullptr};   ///< flip array with values of 1 and -1 assigned by random

      // MemoryHandler mem_; ///< Device memory manager object
  };
}
