/**
 * @file RandomSketchingFWHTHip.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of RandomSketchingFWHTHip class.
 * 
 * @copyright Copyright (c) 2024
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
   * @brief Fast Walsh-Hadamard transform implementation using HIP backend.
   * 
   */
  class RandomSketchingFWHTHip : public RandomSketchingImpl
  {

    using vector_type = vector::Vector;
    public: 
      RandomSketchingFWHTHip();
      virtual ~RandomSketchingFWHTHip();

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output);

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k);
      virtual int reset();

    private:
      index_type n_{0};      ///< size of base vector
      index_type k_rand_{0}; ///< size of sketched vector

      index_type* h_seq_{nullptr};  ///< auxiliary variable used for Fisher-Yates algorithm 
      index_type* h_D_{nullptr};    ///< D is a diagonal matrix (FWHT computed y = PHDx), we store it as an array. D consists of _1_s and _-1_s
      index_type* h_perm_{nullptr}; ///< permuation array, containing _k_ values in range of _0_ to _n-1_

      index_type* d_D_{nullptr};    ///< device mirror of D
      index_type* d_perm_{nullptr}; ///< device mirror of h_perm 
      real_type*  d_aux_{nullptr};  ///< auxiliary variable needed to store partial results in FWHT application.
      
      index_type N_{0};           ///< padded vector size
      index_type log2N_{0};       ///< log2 of N_, used multiple times so we store it
      real_type one_over_k_{0.0}; ///< 1/k, used many times for scaling so we store the value to avoid recomputation

      MemoryHandler mem_; ///< Device memory manager object
  };
}
