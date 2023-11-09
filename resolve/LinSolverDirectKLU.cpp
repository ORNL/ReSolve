#include <cstring> // includes memcpy
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include "LinSolverDirectKLU.hpp"

namespace ReSolve 
{
  using out = io::Logger;

  LinSolverDirectKLU::LinSolverDirectKLU()
  {
    Symbolic_ = nullptr;
    Numeric_ = nullptr;

    L_ = nullptr;
    U_ = nullptr;

    klu_defaults(&Common_) ;
  } 

  LinSolverDirectKLU::~LinSolverDirectKLU()
  {
    klu_free_symbolic(&Symbolic_, &Common_);
    klu_free_numeric(&Numeric_, &Common_);
  }

  int LinSolverDirectKLU::setup(matrix::Sparse* A,
                                matrix::Sparse* /* L */,
                                matrix::Sparse* /* U */,
                                index_type*     /* P */,
                                index_type*     /* Q */,    
                                vector_type*  /* rhs */)
  {
    this->A_ = A;
    return 0;
  }

  void LinSolverDirectKLU::setupParameters(int ordering, double KLU_threshold, bool halt_if_singular) 
  {
    Common_.btf  = 0;
    Common_.ordering = ordering;
    Common_.tol = KLU_threshold;
    Common_.scale = -1;
    Common_.halt_if_singular = halt_if_singular;
  }

  int LinSolverDirectKLU::setParameters() 
  {
    Common_.btf  = 0;
    Common_.ordering = ordering_;
    Common_.tol = pivotThreshold_;
    Common_.scale = -1;
    Common_.halt_if_singular = haltIfSingular_;

    out::summary() << "KLU parameters set:\n"
                   << "\tbtf              = " << Common_.btf              << "\n"
                   << "\tordering         = " << Common_.ordering         << "\n"
                   << "\tpivot threshold  = " << Common_.tol              << "\n"
                   << "\tscale            = " << Common_.scale            << "\n"
                   << "\thalt if singular = " << Common_.halt_if_singular << "\n";
    return 0;
  }

  int LinSolverDirectKLU::analyze() 
  {
    // in case we called this function AGAIN 
    if (Symbolic_ != nullptr) {
      klu_free_symbolic(&Symbolic_, &Common_);
    }

    Symbolic_ = klu_analyze(A_->getNumRows(), A_->getRowData(memory::HOST), A_->getColData(memory::HOST), &Common_) ;

    factors_extracted_ = false;
    if (L_ != nullptr) {
      delete L_; 
      L_ = nullptr;
    }
   
    if (U_ != nullptr) {
      delete U_; 
      U_ = nullptr;
    }

    if (Symbolic_ == nullptr){
      printf("Symbolic_ factorization crashed withCommon_.status = %d \n", Common_.status);
      return 1;
    }
    return 0;
  }

  int LinSolverDirectKLU::factorize() 
  {
    if (Numeric_ != nullptr) {
      klu_free_numeric(&Numeric_, &Common_);
    }

    Numeric_ = klu_factor(A_->getRowData(memory::HOST), A_->getColData(memory::HOST), A_->getValues(memory::HOST), Symbolic_, &Common_);

    factors_extracted_ = false;

    if (L_ != nullptr) {
      delete L_; 
      L_ = nullptr;
    }
    
    if (U_ != nullptr) {
      delete U_; 
      U_ = nullptr;
    }

    if (Numeric_ == nullptr){
      return 1;
    }
    return 0;
  }

  int  LinSolverDirectKLU::refactorize() 
  {
    int kluStatus = klu_refactor (A_->getRowData(memory::HOST), A_->getColData(memory::HOST), A_->getValues(memory::HOST), Symbolic_, Numeric_, &Common_);

    factors_extracted_ = false;

    if (L_ != nullptr) {
      delete L_; 
      L_ = nullptr;
    }
   
    if (U_ != nullptr) {
      delete U_; 
      U_ = nullptr;
    }

    if (!kluStatus){
      //display error
      return 1;
    }
    return 0;
  }

  int LinSolverDirectKLU::solve(vector_type* rhs, vector_type* x) 
  {
    //copy the vector

    //  std::memcpy(x, rhs, A->getNumRows() * sizeof(real_type));

    x->update(rhs->getData(memory::HOST), memory::HOST, memory::HOST);
    x->setDataUpdated(memory::HOST);

    int kluStatus = klu_solve(Symbolic_, Numeric_, A_->getNumRows(), 1, x->getData(memory::HOST), &Common_);

    if (!kluStatus){
      return 1;
    }
    return 0;
  }

  matrix::Sparse* LinSolverDirectKLU::getLFactor()
  {
    if (!factors_extracted_) {
      const int nnzL = Numeric_->lnz;
      const int nnzU = Numeric_->unz;

      L_ = new matrix::Csc(A_->getNumRows(), A_->getNumColumns(), nnzL);
      U_ = new matrix::Csc(A_->getNumRows(), A_->getNumColumns(), nnzU);
      L_->allocateMatrixData(memory::HOST);
      U_->allocateMatrixData(memory::HOST);
    
      int ok = klu_extract(Numeric_, 
                           Symbolic_, 
                           L_->getColData(memory::HOST), 
                           L_->getRowData(memory::HOST), 
                           L_->getValues( memory::HOST), 
                           U_->getColData(memory::HOST), 
                           U_->getRowData(memory::HOST), 
                           U_->getValues( memory::HOST), 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr,
                           nullptr,
                           nullptr,
                           &Common_);

      L_->setUpdated(memory::HOST);
      U_->setUpdated(memory::HOST);
      (void) ok; // TODO: Check status in ok before setting `factors_extracted_`
      factors_extracted_ = true;
    }
    return L_;
  }

  matrix::Sparse* LinSolverDirectKLU::getUFactor()
  {
    if (!factors_extracted_) {
      const int nnzL = Numeric_->lnz;
      const int nnzU = Numeric_->unz;

      L_ = new matrix::Csc(A_->getNumRows(), A_->getNumColumns(), nnzL);
      U_ = new matrix::Csc(A_->getNumRows(), A_->getNumColumns(), nnzU);
      L_->allocateMatrixData(memory::HOST);
      U_->allocateMatrixData(memory::HOST);
      int ok = klu_extract(Numeric_, 
                           Symbolic_, 
                           L_->getColData(memory::HOST), 
                           L_->getRowData(memory::HOST), 
                           L_->getValues( memory::HOST), 
                           U_->getColData(memory::HOST), 
                           U_->getRowData(memory::HOST), 
                           U_->getValues( memory::HOST), 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr,
                           nullptr,
                           nullptr,
                           &Common_);

      L_->setUpdated(memory::HOST);
      U_->setUpdated(memory::HOST);

      (void) ok; // TODO: Check status in ok before setting `factors_extracted_`
      factors_extracted_ = true;
    }
    return U_;
  }

  index_type* LinSolverDirectKLU::getPOrdering()
  {
    if (Numeric_ != nullptr){
      P_ = new index_type[A_->getNumRows()];
      size_t nrows = static_cast<size_t>(A_->getNumRows());
      std::memcpy(P_, Numeric_->Pnum, nrows * sizeof(index_type));
      return P_;
    } else {
      return nullptr;
    }
  }


  index_type* LinSolverDirectKLU::getQOrdering()
  {
    if (Numeric_ != nullptr){
      Q_ = new index_type[A_->getNumRows()];
      size_t nrows = static_cast<size_t>(A_->getNumRows());
      std::memcpy(Q_, Symbolic_->Q, nrows * sizeof(index_type));
      return Q_;
    } else {
      return nullptr;
    }
  }
}
