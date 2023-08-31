#pragma once
#include "Common.hpp"
#include <resolve/vector/VectorHandler.hpp>
#include <iostream>
#include <cassert>
namespace ReSolve 
{
  enum GS_variant { mgs, cgs2, mgs_two_synch, mgs_pm, cgs};
  class GramSchmidt
  {
    using vector_type = vector::Vector;
    public:
    GramSchmidt();
    GramSchmidt(VectorHandler* vh, GS_variant variant);
    ~GramSchmidt();
    int setVariant(GS_variant variant);
    GS_variant  getVariant();
    real_type* getL(); //only for low synch, returns null ptr otherwise 

    int setup(index_type restart);
    int orthogonalize(index_type n, real_type* V, real_type* H, index_type i, std::string memspace);

    private:
    GS_variant variant_;
    bool setup_complete_; //to avoid double allocations and stuff

    index_type num_vecs_; //the same as restart  
    real_type* d_rvGPU_;
    real_type* d_Hcolumn_;
    real_type* d_H_col_;

    real_type* h_L_;
    real_type* h_rv_;
    real_type* h_aux_;
    VectorHandler* vector_handler_;

    real_type one_ = 1.0;
    real_type minusone_ = -1.0;
    real_type zero_ = 0.0; 
  };

}//namespace
