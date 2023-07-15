#include "LinSolverDirectKLU.hpp"

namespace ReSolve 
{
  LinSolverDirectKLU::LinSolverDirectKLU()
  {
    Symbolic_ = nullptr;
    Numeric_ = nullptr;
    klu_defaults(&Common_) ;
  } 

  LinSolverDirectKLU::~LinSolverDirectKLU()
  {
    klu_free_symbolic(&Symbolic_, &Common_);
    klu_free_numeric(&Numeric_, &Common_);
  }

  int LinSolverDirectKLU::setup(Matrix* A)
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

  int LinSolverDirectKLU::analyze() 
  {
    Symbolic_ = klu_analyze(A_->getNumRows(), A_->getRowData("cpu"), A_->getColData("cpu"), &Common_) ;

    if (Symbolic_ == nullptr){
      printf("Symbolic_ factorization crashed withCommon_.status = %d \n", Common_.status);
      return 1;
    }
    return 0;
  }

  int LinSolverDirectKLU::factorize() 
  {
    Numeric_ = klu_factor(A_->getRowData("cpu"), A_->getColData("cpu"),A_->getValues("cpu"), Symbolic_, &Common_);

    if (Numeric_ == nullptr){
      return 1;
    }
    return 0;
  }

  int  LinSolverDirectKLU::refactorize() 
  {
    int kluStatus = klu_refactor (A_->getRowData("cpu"), A_->getColData("cpu"), A_->getValues("cpu"), Symbolic_, Numeric_, &Common_);

    if (!kluStatus){
      //display error
      return 1;
    }
    return 0;
  }

  int LinSolverDirectKLU::solve(Vector* rhs, Vector* x) 
  {
    //copy the vector

    //  std::memcpy(x, rhs, A->getNumRows() * sizeof(Real));

    x->update(rhs->getData("cpu"), "cpu", "cpu");
    x->setDataUpdated("cpu");

    int kluStatus = klu_solve(Symbolic_, Numeric_, A_->getNumRows(), 1, x->getData("cpu"), &Common_);

    if (!kluStatus){
      return 1;
    }
    return 0;
  }

  Matrix* LinSolverDirectKLU::getLFactor()
  {
    if (!factors_extracted_) {
      const int nnzL = Numeric_->lnz;
      const int nnzU = Numeric_->unz;

      L_ = new MatrixCSC(A_->getNumRows(), A_->getNumColumns(), nnzL);
      U_ = new MatrixCSC(A_->getNumRows(), A_->getNumColumns(), nnzU);
      L_->allocateMatrixData("cpu");
      U_->allocateMatrixData("cpu");
      int ok = klu_extract(Numeric_, 
                           Symbolic_, 
                           L_->getColData("cpu"), 
                           L_->getRowData("cpu"), 
                           L_->getValues("cpu"), 
                           U_->getColData("cpu"), 
                           U_->getRowData("cpu"), 
                           U_->getValues("cpu"), 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr,
                           nullptr,
                           nullptr,
                           &Common_);

      L_->setUpdated("cpu");
      U_->setUpdated("cpu");

      factors_extracted_ = true;
    }
    return L_;
  }

  Matrix* LinSolverDirectKLU::getUFactor()
  {
    if (!factors_extracted_) {
      const int nnzL = Numeric_->lnz;
      const int nnzU = Numeric_->unz;

      L_ = new MatrixCSC(A_->getNumRows(), A_->getNumColumns(), nnzL);
      U_ = new MatrixCSC(A_->getNumRows(), A_->getNumColumns(), nnzU);
      L_->allocateMatrixData("cpu");
      U_->allocateMatrixData("cpu");
      int ok = klu_extract(Numeric_, 
                           Symbolic_, 
                           L_->getColData("cpu"), 
                           L_->getRowData("cpu"), 
                           L_->getValues("cpu"), 
                           U_->getColData("cpu"), 
                           U_->getRowData("cpu"), 
                           U_->getValues("cpu"), 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr, 
                           nullptr,
                           nullptr,
                           nullptr,
                           &Common_);

      L_->setUpdated("cpu");
      U_->setUpdated("cpu");
      factors_extracted_ = true;
    }
    return U_;
  }

  Int* LinSolverDirectKLU::getPOrdering()
  {
    if (Numeric_ != nullptr){
      P_ = new Int[A_->getNumRows()];
      std::memcpy(P_, Numeric_->Pnum, A_->getNumRows() * sizeof(Int));
      return P_;
    } else {
      return nullptr;
    }
  }


  Int* LinSolverDirectKLU::getQOrdering()
  {
    if (Numeric_ != nullptr){
      Q_ = new Int[A_->getNumRows()];
      std::memcpy(Q_, Symbolic_->Q, A_->getNumRows() * sizeof(Int));
      return Q_;
    } else {
      return nullptr;
    }
  }
}
