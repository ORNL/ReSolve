#include <resolve/hykkt/PermClass.hpp>
#include <resolve/hykkt/cpuHykktPermutationKernels.hpp>
#include <cstdio>
#include "amd.h"

// Creates a class for the permutation of $H_\gamma$ in (6)
PermClass::PermClass(int n_h, int nnz_h, int nnz_j) 
  : n_h_(n_h),
    nnz_h_(nnz_h),
    nnz_j_(nnz_j)
  {
    allocateWorkspace();
  }

  PermClass::~PermClass()
  {
    if(perm_is_default_){
      delete [] perm_;
    }
    delete [] rev_perm_;
    delete [] perm_map_h_;
    delete [] perm_map_j_;
    delete [] perm_map_jt_;
  }

  void PermClass::addHInfo(int* h_i, int* h_j)
  {
    h_i_ = h_i;
    h_j_ = h_j;
  }
  
  void PermClass::addJInfo(int* j_i, int* j_j, int n_j, int m_j)
  {
    j_i_ = j_i;
    j_j_ = j_j;
    n_j_ = n_j;
    m_j_ = m_j;
  }
  
  void PermClass::addJtInfo(int* jt_i, int* jt_j)
  {
    jt_i_ = jt_i;
    jt_j_ = jt_j;
  }
  
  void PermClass::addPerm(int* custom_perm)
  {
    perm_is_default_ = false;
    perm_ = custom_perm;
  }
 
// symAmd permutation of $H_\gamma$ in (6)
  void PermClass::symAmd()
  {
    double Control[AMD_CONTROL], Info[AMD_INFO];
	
    amd_defaults(Control);
    amd_control(Control);
	
    int result = amd_order(n_h_, h_i_, h_j_, perm_, Control, Info);
	
    if (result != AMD_OK)
    {
       printf("AMD failed\n");
       exit(1);
    }
  }
  
  void PermClass::invertPerm()
  {
    reverse_perm(n_h_, perm_, rev_perm_);
  }

  void PermClass::vecMapRC(int* b_i, int* b_j)
  {
    make_vecMapRC(n_h_, h_i_, h_j_, perm_, rev_perm_, b_i, b_j, perm_map_h_);
  }

  void PermClass::vecMapC(int* b_j)
  {
    make_vecMapC(n_j_, j_i_, j_j_, rev_perm_, b_j, perm_map_j_);
  }

  void PermClass::vecMapR(int* b_i, int* b_j)
  {
    make_vecMapR(m_j_, jt_i_, jt_j_, perm_, b_i, b_j, perm_map_jt_);
  }
  
  void PermClass::map_index(PermutationType permutation,
      double* old_val,
      double* new_val)
  {
    switch(permutation)
    {
      case perm_v: 
        cpu_map_idx(n_h_, perm_, old_val, new_val);
        break;
      case rev_perm_v: 
        cpu_map_idx(n_h_, rev_perm_, old_val, new_val);
        break;
      case perm_h_v: 
        cpu_map_idx(nnz_h_, perm_map_h_, old_val, new_val);
        break;
      case perm_j_v: 
        cpu_map_idx(nnz_j_, perm_map_j_, old_val, new_val);
        break;
      case perm_jt_v: 
        cpu_map_idx(nnz_j_, perm_map_jt_, old_val, new_val);
        break;
      default:
        printf("Valid arguments are perm_v, rev_perm_v, perm_h_v, perm_j_v, perm_jt_v\n");
    }
  }
  
  void PermClass::allocateWorkspace()
  {
    perm_ = new int[n_h_];
    rev_perm_ = new int[n_h_];
    perm_map_h_ = new int[nnz_h_];
    perm_map_j_ = new int[nnz_j_];
    perm_map_jt_ = new int[nnz_j_];
  }
