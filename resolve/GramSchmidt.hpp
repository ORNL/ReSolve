#pragma once

#include <iostream>
#include <cassert>

#include "Common.hpp"
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve 
{
  class GramSchmidt
  {
    private:
      using vector_type = vector::Vector;

    public:
      enum GSVariant {MGS = 0, 
                      CGS2,
                      MGS_TWO_SYNC, 
                      MGS_PM,
                      CGS1};
      
      GramSchmidt() = delete;
      GramSchmidt(VectorHandler* vh, GSVariant variant);
      ~GramSchmidt();
      int setVariant(GramSchmidt::GSVariant variant);
      GSVariant  getVariant();
      real_type* getL(); //only for low synch, returns null ptr otherwise 

      int setup(index_type n, index_type restart);
      int orthogonalize(index_type n, vector_type* V, real_type* H, index_type i);
      bool isSetupComplete();

    private:
      int freeGramSchmidtData();
    
      GSVariant variant_{MGS};
      bool setup_complete_{false}; //to avoid double allocations

      index_type num_vecs_; //the same as restart  
      vector_type* vec_rv_{nullptr};
      vector_type* vec_Hcolumn_{nullptr};

      real_type* h_L_{nullptr};
      real_type* h_aux_{nullptr};
      VectorHandler* vector_handler_{nullptr};

      vector_type* vec_v_{nullptr}; // aux variable
      vector_type* vec_w_{nullptr}; // aux variable
    
      MemoryHandler mem_; ///< Device memory manager object
      memory::MemorySpace memspace_;
  };

} // namespace ReSolve
