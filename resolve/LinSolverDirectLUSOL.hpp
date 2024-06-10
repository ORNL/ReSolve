#pragma once

#include <resolve/Common.hpp>
#include <resolve/LinSolver.hpp>

#include <cstdint>
#include <memory>

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

  class LinSolverDirectLUSOL : public LinSolverDirect
  {
    using vector_type = vector::Vector;

    public:
      LinSolverDirectLUSOL();
      ~LinSolverDirectLUSOL();

      int setup(matrix::Sparse* A,
                matrix::Sparse* L = nullptr,
                matrix::Sparse* U = nullptr,
                index_type*     P = nullptr,
                index_type*     Q = nullptr,
                vector_type*  rhs = nullptr) override;

      int analyze() override;
      int factorize() override;
      int refactorize() override;
      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* x) override;

      matrix::Sparse* getLFactor() override;
      matrix::Sparse* getUFactor() override;
      index_type* getPOrdering() override;
      index_type* getQOrdering() override;

      virtual void setPivotThreshold(real_type tol) override;
      virtual void setOrdering(int ordering) override;
      virtual void setHaltIfSingular(bool isHalt) override;

      virtual real_type getMatrixConditionNumber() override;

    private:
      //NOTE: a_, indc_, and indr_ may need to be passed along to the GPU at some
      //      point in the future, so they are manually managed

      /// @brief Storage used for the matrices
      ///
      /// Primary storage area used by LUSOL. Used to hold the nonzeros of matrices
      /// passed along API boundaries and as a general scratch region
      real_type* a_ = nullptr;

      /// @brief Row indices of matrices passed along API boundaries
      index_type* indc_ = nullptr;

      /// @brief Column indices of matrices passed along API boundaries
      index_type* indr_ = nullptr;

      /// @brief The number of nonzero elements within the input matrix, A
      index_type nelem_ = -1;

      /// @brief The length of the dynamically-allocated arrays held within `a_`,
      ///        `indc_`, and `indr_`
      ///
      /// This should be much greater than the number of nonzeroes in the input
      /// matrix A, and a (hopefully optimal enough for most cases) formula derived
      /// from correspondence with Michael Saunders is
      ///
      /// ```cpp
      /// //NOTE: parmlu_[7] specifies the threshold at which LUSOL uses a dense LU
      /// //      factorization, and is referred to within the source as "dens2"
      /// if (nelem_ >= parmlu_[7] * m_ * n_) {
      ///   // A is dense
      ///   lena_ = m_ * n_;
      /// } else {
      ///   // A is sparse
      ///   lena_ = min(5 * nelem_, 2 * m_ * n_);
      /// }
      /// ```
      ///
      /// The idea behind the approximate bound `5 * nelem_` is that the LU factors are
      /// almost always going to be significantly more dense compared to the density of
      /// the input matrix A. However, there are cases with smaller (but still sparse)
      /// matrices in which `2 * m_ * n_` is likely preferable.
      ///
      /// In the dense case, predicated upon the number of nonzeroes exceeding
      /// `parmlu_[7] * m_ * n_` (the threshold at which LUSOL uses a dense LU
      /// factorization) is the use of the approximate bound `m_ * n_`, which should
      /// always be enough.
      ///
      /// Note that this is not an upper bound on the required space; the size of this
      /// buffer may be insufficient, in which case a call to a LUSOL subroutine
      /// utilizing it will return with inform set to 7, and the intended behavior of
      /// the callee is that they should resize `a_`, `indc_`, and `indr_` to at least
      /// the value specified in `luparm_[12]`
      index_type lena_ = -1;

      /// @brief The number of rows in the input matrix, A
      index_type m_ = -1;

      /// @brief The number of columns in the input matrix, A
      index_type n_ = -1;

      /// @brief Index-typed parameters passed along the API boundary
      index_type luparm_[30] = {0};

      /// @brief Real-typed parameters passed along the API boundary
      real_type parmlu_[30] = {0};

      /// @brief The row permutation
      index_type* p_ = nullptr;

      /// @brief The column permutation
      index_type* q_ = nullptr;

      /// @brief Number of entries within nontrivial columns of L, stored in pivot order
      index_type* lenc_ = nullptr;

      /// @brief Number of entries in each row of U, stored in original order
      index_type* lenr_ = nullptr;

      /// @brief Appears to be internal storage for LUSOL, used by the LU update routines
      index_type* locc_ = nullptr;

      /// @brief Points to the beginning of rows of U within a
      index_type* locr_ = nullptr;

      //TODO: it would be nice to have more information about these "undefined" (as
      //      said within the source code documentation of lu1fac) parameters
      //
      //      there is some amount of documentation in the "notes on array names"
      //      section, but given they're only really storage parameters and aren't
      //      useful post-factorization, we'll leave it at "undefined" for now

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
}
