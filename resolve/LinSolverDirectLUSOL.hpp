#pragma once

#include <cstdint>
#include <memory>

#include <resolve/Common.hpp>
#include <resolve/LinSolverDirect.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }

  namespace matrix
  {
    class Sparse;
  }

  /**
   * @brief Wrapper for LUSOL solver.
   * 
   * LUSOL Fortran code is in file `lusol.f90`.
   * 
   */
  class LinSolverDirectLUSOL : public LinSolverDirect
  {
      using vector_type = vector::Vector;

    public:
      LinSolverDirectLUSOL();
      ~LinSolverDirectLUSOL();

      /// @brief Setup function of the linear solver
      int setup(matrix::Sparse* A,
                matrix::Sparse* L = nullptr,
                matrix::Sparse* U = nullptr,
                index_type*     P = nullptr,
                index_type*     Q = nullptr,
                vector_type*  rhs = nullptr) override;

      /// @brief Analysis function of LUSOL
      int analyze() override;

      int factorize() override;
      int refactorize() override;
      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* x) override;

      /// @brief Returns the L factor of the solution in CSC format
      matrix::Sparse* getLFactor() override;

      /// @brief Returns the U factor of the solution in CSR format
      matrix::Sparse* getUFactor() override;

      index_type* getPOrdering() override;
      index_type* getQOrdering() override;

      int setCliParam(const std::string id, const std::string value) override;
      std::string getCliParamString(const std::string id) const override;
      index_type getCliParamInt(const std::string id) const override;
      real_type getCliParamReal(const std::string id) const override;
      bool getCliParamBool(const std::string id) const override;
      int printCliParam(const std::string id) const override;

    private:
      int allocateSolverData();
      int freeSolverData();

      bool is_solver_data_allocated_{false};

      MemoryHandler mem_;

      /// @brief Indicates if we have factorized the matrix yet
      bool is_factorized_ = false;

      /// @brief Storage used for the matrices
      ///
      /// Primary workspace used by LUSOL. Used to hold the nonzeros of matrices
      /// passed along API boundaries and as a general scratch region
      real_type* a_ = nullptr;

      /// @brief Row data of matrices passed along API boundaries, in addition to
      ///        functioning as additional workspace storage for LUSOL
      index_type* indc_ = nullptr;

      /// @brief Column data of matrices passed along API boundaries, in addition to
      ///        functioning as additional workspace storage for LUSOL
      index_type* indr_ = nullptr;

      /// @brief The number of nonzero elements within the input matrix, A
      index_type nelem_ = 0;

      /// @brief The permutation vector P, stored in the way LUSOL expects it to be (1-indexed)
      index_type* p_ = nullptr;

      /// @brief The permutation vector Q, stored in the way LUSOL expects it to be (1-indexed)
      index_type* q_ = nullptr;

      /// @brief The length of the dynamically-allocated arrays held within `a_`,
      ///        `indc_`, and `indr_`
      ///
      /// This should be much greater than the number of nonzeroes in the input
      /// matrix A, as stated in LUSOL's source code.
      ///
      /// Note that this is not an upper bound on the required space; the size of this
      /// buffer may be insufficient, in which case a call to a LUSOL subroutine
      /// utilizing it will return with inform set to 7, and the intended behavior of
      /// the callee is that they should resize `a_`, `indc_`, and `indr_` to at least
      /// the value specified in `luparm_[12]`
      index_type lena_ = 0;

      /// @brief The number of rows in the input matrix, A
      index_type m_ = 0;

      /// @brief The number of columns in the input matrix, A
      index_type n_ = 0;

      /// @brief Index-typed parameters passed along the API boundary
      index_type luparm_[30] = {0};

      /// @brief Real-typed parameters passed along the API boundary
      real_type parmlu_[30] = {0};

      /// @brief Number of entries within nontrivial columns of L, stored in pivot order
      index_type* lenc_ = nullptr;

      /// @brief Number of entries in each row of U, stored in original order
      index_type* lenr_ = nullptr;

      /// @brief Appears to be internal storage for LUSOL, used by the LU update routines
      index_type* locc_ = nullptr;

      /// @brief Points to the beginning of rows of U within a
      index_type* locr_ = nullptr;

      // TODO: it would be nice to have more information about these "undefined" (as
      //       said within the source code documentation of lu1fac) parameters
      //
      //       there is some amount of documentation in the "notes on array names"
      //       section, but given they're only really storage parameters and aren't
      //       useful post-factorization, we'll leave it at "undefined" for now

      /// @brief Undefined value
      index_type* iploc_ = nullptr;

      /// @brief Undefined value
      index_type* iqloc_ = nullptr;

      /// @brief Undefined value
      index_type* ipinv_ = nullptr;

      /// @brief Undefined value
      index_type* iqinv_ = nullptr;

      /// @brief Indicates singularity during LU factorization, otherwise contains either
      ///        the solution or target for solving a linear system
      ///
      /// Generally speaking, `w_[j] == +max(jth column of U)`, but if the
      /// `j`th column is a singularity, `w_[j] == -max(jth column of U)`. Hence,
      /// `w_[j] <= 0` implies that the column `j` of A is likely dependent on the
      /// other columns of A.
      ///
      /// When solving a linear system `A*w_ = v_`, `w_` contains the solution. It is not
      /// important what `w_` contains prior to this.
      real_type* w_ = nullptr;
  };
} // namespace ReSolve
