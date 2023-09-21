#pragma once
#include "Common.hpp"
#include <resolve/vector/VectorHandler.hpp>
#include <iostream>
#include <cassert>
namespace ReSolve 
{
  enum GSVariant { mgs = 0, 
                   cgs2 = 1,
                   mgs_two_synch = 2, 
                   mgs_pm = 3, 
                   cgs1 = 4 };
  class GramSchmidt
  {
    using vector_type = vector::Vector;
    public:
    GramSchmidt();
    GramSchmidt(VectorHandler* vh, GSVariant variant);
    ~GramSchmidt();
    int setVariant(GSVariant variant);
    GSVariant  getVariant();
    real_type* getL(); //only for low synch, returns null ptr otherwise 

    int setup(index_type n, index_type restart);
    int orthogonalize(index_type n, vector_type* V, real_type* H, index_type i, std::string memspace);

    private:
    
    GSVariant variant_;
    bool setup_complete_; //to avoid double allocations and stuff

    index_type num_vecs_; //the same as restart  
    vector_type* vec_rv_;
    vector_type* vec_Hcolumn_;
//    vector_type* d_H_col_;

    real_type* h_L_;
    real_type* h_rv_;
    real_type* h_aux_;
    VectorHandler* vector_handler_;

    vector_type* vec_v_; // aux variable
    vector_type* vec_w_; // aux variable
  };

}//namespace
