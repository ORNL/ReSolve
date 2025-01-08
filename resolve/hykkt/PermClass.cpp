#include <resolve/hykkt/PermClass.hpp>
#include <resolve/hykkt/cpuHykktPermutationKernels.hpp>
#include <cstdio>
#include "amd.h"
namespace ReSolve::Hykkt
{
  // Creates a class for the permutation of $H_\gamma$ in (6)
  PermClass::PermClass(int n_hes, int nnz_hes, int nnz_jac) 
  : n_hes_(n_hes),
    nnz_hes_(nnz_hes),
    nnz_jac_(nnz_jac)
  {
    allocateWorkspace();
  }

  PermClass::~PermClass()
  {
    if(perm_is_default_){
      delete [] perm_;
    }
    delete [] rev_perm_;
    delete [] perm_map_hes_;
    delete [] perm_map_jac_;
    delete [] perm_map_jac_tr_;
  }

  void PermClass::addHInfo(int* hes_i, int* hes_j)
  {
    hes_i_ = hes_i;
    hes_j_ = hes_j;
  }
  
  void PermClass::addJInfo(int* jac_i, int* jac_j, int n_jac, int m_jac)
  {
    jac_i_ = jac_i;
    jac_j_ = jac_j;
    n_jac_ = n_jac;
    m_jac_ = m_jac;
  }
  
  void PermClass::addJtInfo(int* jac_tr_i, int* jac_tr_j)
  {
    jac_tr_i_ = jac_tr_i;
    jac_tr_j_ = jac_tr_j;
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
	
    int result = amd_order(n_hes_, hes_i_, hes_j_, perm_, Control, Info);
	
    if (result != AMD_OK)
    {
       printf("AMD failed\n");
       exit(1);
    }
  }
  
  void PermClass::invertPerm()
  {
    reversePerm(n_hes_, perm_, rev_perm_);
  }

  void PermClass::vecMapRC(int* rhs_i, int* rhs_j)
  {
    makeVecMapRC(n_hes_, hes_i_, hes_j_, perm_, rev_perm_, rhs_i, rhs_j, perm_map_hes_);
  }

  void PermClass::vecMapC(int* rhs_j)
  {
    makeVecMapC(n_jac_, jac_i_, jac_j_, rev_perm_, rhs_j, perm_map_jac_);
  }

  void PermClass::vecMapR(int* rhs_i, int* rhs_j)
  {
    makeVecMapR(m_jac_, jac_tr_i_, jac_tr_j_, perm_, rhs_i, rhs_j, perm_map_jac_tr_);
  }
  
  void PermClass::map_index(PermutationType permutation,
      double* old_val,
      double* new_val)
  {
    switch(permutation)
    {
      case PERM_V: 
        cpuMapIdx(n_hes_, perm_, old_val, new_val);
        break;
      case REV_PERM_V: 
        cpuMapIdx(n_hes_, rev_perm_, old_val, new_val);
        break;
      case PERM_HES_V: 
        cpuMapIdx(nnz_hes_, perm_map_hes_, old_val, new_val);
        break;
      case PERM_JAC_V: 
        cpuMapIdx(nnz_jac_, perm_map_jac_, old_val, new_val);
        break;
      case PERM_JAC_TR_V: 
        cpuMapIdx(nnz_jac_, perm_map_jac_tr_, old_val, new_val);
        break;
      default:
        printf("Valid arguments are PERM_V, REV_PERM_V, PERM_H_V, PERM_J_V, PERM_JT_V\n");
    }
  }
  
  void PermClass::allocateWorkspace()
  {
    perm_ = new int[n_hes_];
    rev_perm_ = new int[n_hes_];
    perm_map_hes_ = new int[nnz_hes_];
    perm_map_jac_ = new int[nnz_jac_];
    perm_map_jac_tr_ = new int[nnz_jac_];
  }
}
