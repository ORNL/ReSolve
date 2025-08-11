#include "CholeskySolverHip.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    CholeskySolverHip::CholeskySolverHip()
    {
      rocblas_create_handle(&handle_);
      rocsolver_create_rfinfo(&rfinfo_, handle_);
      rocsolver_set_rfinfo_mode(rfinfo_, rocsolver_rfinfo_mode_cholesky);

      cholmod_start(&Common_);

      Common_.nmethods = 1;
      // Use natural ordering
      Common_.method[0].ordering = CHOLMOD_NATURAL;
      Common_.final_ll = true;

      A_chol_        = nullptr;
      factorization_ = nullptr;
      L_ = nullptr;
      Q_ = nullptr;
    }

    CholeskySolverHip::~CholeskySolverHip()
    {
      rocsolver_destroy_rfinfo(rfinfo_);
      rocblas_destroy_handle(handle_);
      if (A_chol_)
      {
        cholmod_free_sparse(&A_chol_, &Common_);
      }
      if (factorization_)
      {
        cholmod_free_factor(&factorization_, &Common_);
      }
      cholmod_finish(&Common_);
      free(L_);
      free(Q_);
    }

    void CholeskySolverHip::addMatrixInfo(matrix::Csr* A)
    {
      // free previous members if they exist
      // store A as cholmod_sparse
      if (A_chol_)
      {
        cholmod_free_sparse(&A_chol_, &Common_);
      }
      A_chol_ = convertToCholmod(A);
      A_ = A;
    }

    void CholeskySolverHip::symbolicAnalysis()
    {
      // cholmod analyze
      factorization_ = cholmod_analyze(A_chol_, &Common_);
      if (Common_.status < 0)
      {
        out::error() << "Cholesky symbolic analysis failed with status: " << Common_.status << "\n";
      }
    }

    void CholeskySolverHip::numericalFactorization(real_type tol)
    {
      (void) tol; // Mark tol as unused

      if (!L_) // is initial factorization
      {
        // Initial factorization on CPU
        cholmod_factorize(A_chol_, factorization_, &Common_);
        if (Common_.status < 0)
        {
          out::error() << "Cholesky factorization failed with status: " << Common_.status << "\n";
        }

        // Extract initial factorization to L_
        cholmod_sparse* L_chol = cholmod_factor_to_sparse(factorization_, &Common_);
        cholmod_sparse* L_chol_tr = cholmod_transpose(L_chol, 1, &Common_);
        L_ = new matrix::Csr((index_type) L_chol->nrow, (index_type) L_chol->ncol, (index_type) L_chol->nzmax);
        L_->allocateMatrixData(memory::DEVICE);
        L_->copyDataFrom(static_cast<index_type*>(L_chol_tr->p),
                         static_cast<index_type*>(L_chol_tr->i),
                         static_cast<real_type*>(L_chol_tr->x),
                         memory::HOST,
                         memory::DEVICE);
        // L_->syncData(memory::HOST);
        // L_->print(std::cout);
        // Store fill-in reducing permutation.
        // Within HyKKT, this will be the identity permutation because the Permutation class will permute the matrix.
        mem_.allocateArrayOnDevice(&Q_, A_->getNumRows());
        mem_.copyArrayHostToDevice(Q_, static_cast<index_type*>(factorization_->Perm), A_->getNumRows());

        // Store analysis in rfinfo_
        rocblas_status status = rocsolver_dcsrrf_analysis(handle_,
                                  A_->getNumRows(),
                                  0,
                                  A_->getNnz(),
                                  A_->getRowData(memory::DEVICE),
                                  A_->getColData(memory::DEVICE),
                                  A_->getValues(memory::DEVICE),
                                  L_->getNnz(),
                                  L_->getRowData(memory::DEVICE),
                                  L_->getColData(memory::DEVICE),
                                  L_->getValues(memory::DEVICE),
                                  nullptr,
                                  Q_,
                                  nullptr,
                                  A_->getNumRows(),
                                  rfinfo_);
        if (status != rocblas_status_success)
        {
          out::error() << "Analysis step failed with status: " << status << "\n";
        }
      }
      else //re-factorize
      {
        rocblas_status status = rocsolver_dcsrrf_refactchol(handle_,
                                    A_->getNumRows(),
                                    A_->getNnz(),
                                    A_->getRowData(memory::DEVICE),
                                    A_->getColData(memory::DEVICE),
                                    A_->getValues(memory::DEVICE),
                                    L_->getNnz(),
                                    L_->getRowData(memory::DEVICE),
                                    L_->getColData(memory::DEVICE),
                                    L_->getValues(memory::DEVICE),
                                    Q_,
                                    rfinfo_);
        if (status != rocblas_status_success)
        {
          out::error() << "Refactorization step failed with status: " << status << "\n";
        }
        L_->setUpdated(memory::DEVICE);
      }
      
    }

    void CholeskySolverHip::solve(vector::Vector* x, vector::Vector* b)
    {
      x->copyDataFrom(b, memory::DEVICE, memory::DEVICE);
      // uncomment to check that none of these are null
      // std::cout << L_->getNumRows() << "\n";
      // std::cout << L_->getNnz() << "\n";
      // std::cout << L_->getRowData(memory::DEVICE) << "\n";
      // std::cout << L_->getColData(memory::DEVICE) << "\n";
      // std::cout << L_->getValues(memory::DEVICE) << "\n";
      // std::cout << x->getData(memory::DEVICE) << "\n";
      // std::cout << Q_ << "\n";
      // std::cout << x->getSize() << "\n";
      // std::cout << rfinfo_ << "\n";
      rocblas_status status = rocsolver_dcsrrf_solve(handle_,
                             L_->getNumRows(),
                             1,
                             L_->getNnz(),
                             L_->getRowData(memory::DEVICE),
                             L_->getColData(memory::DEVICE),
                             L_->getValues(memory::DEVICE),
                             nullptr,
                             Q_,
                             x->getData(memory::DEVICE),
                             x->getSize(),
                             rfinfo_);
      if (status != rocblas_status_success)
      {
        out::error() << "Direct solve step failed with status: " << status << "\n";
      }
      x->setDataUpdated(memory::DEVICE);
    }

    cholmod_sparse* CholeskySolverHip::convertToCholmod(matrix::Csr* A)
    {
      A_chol_ = cholmod_allocate_sparse((size_t) A->getNumRows(),
                                        (size_t) A->getNumColumns(),
                                        (size_t) A->getNnz(),
                                        1,
                                        1,
                                        1,
                                        CHOLMOD_REAL,
                                        &Common_);
      mem_.copyArrayHostToHost(
          static_cast<int*>(A_chol_->p), A->getRowData(memory::HOST), A->getNumRows() + 1);
      mem_.copyArrayHostToHost(
          static_cast<int*>(A_chol_->i), A->getColData(memory::HOST), A->getNnz());
      mem_.copyArrayHostToHost(
          static_cast<double*>(A_chol_->x), A->getValues(memory::HOST), A->getNnz());

      return A_chol_;
    }
  } // namespace hykkt
} // namespace ReSolve
