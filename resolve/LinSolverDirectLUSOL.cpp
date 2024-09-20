#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "LinSolverDirectLUSOL.hpp"
#include "lusol/lusol.hpp"

#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using out = io::Logger;

  LinSolverDirectLUSOL::LinSolverDirectLUSOL()
  {
    // File number for printed messages
    luparm_[0] = 6;

    // Set LUSOL output print level
    switch (out::verbosity()) {
      case io::Logger::NONE:
        luparm_[1] = -1;
        break;
      case io::Logger::ERRORS:
      case io::Logger::WARNINGS:
        luparm_[1] = 0;
        break;
      case io::Logger::SUMMARY:
        luparm_[1] = 10;
        break;
      case io::Logger::EVERYTHING:
        luparm_[1] = 50;
        break;
    }

    // maximum number of columns searched allowed in a Markowitz-type
    // search for the next pivot element. For some of the factorization, the
    // number of rows searched is maxrow = maxcol - 1.
    luparm_[2] = 5;

    // Pivoting type
    // luparm_[5] = 0    =>  TPP: Threshold Partial   Pivoting.        0
    //            = 1    =>  TRP: Threshold Rook      Pivoting.
    //            = 2    =>  TCP: Threshold Complete  Pivoting.
    //            = 3    =>  TSP: Threshold Symmetric Pivoting.
    //            = 4    =>  TDP: Threshold Diagonal  Pivoting.
    luparm_[5] = 0;

    // Keep factors (1 == keep; 0 == discard)
    luparm_[7] = 1;

    // Max Lij allowed during factor
    parmlu_[0] = 10.0;

    // Max Lij allowed during updates.
    parmlu_[1] = 10.0;

    // Absolute tolerance for treating reals as zero.
    parmlu_[2] = std::pow(std::numeric_limits<real_type>::epsilon(), 0.8);

    // Absolute tol for flagging small diagonals of U.
    parmlu_[3] = std::pow(std::numeric_limits<real_type>::epsilon(), 0.67);

    // Relative tol for flagging small diagonals of U.
    parmlu_[4] = std::pow(std::numeric_limits<real_type>::epsilon(), 0.67);

    // Factor limiting waste space in  U.
    parmlu_[5] = 3.0;

    // The density at which the Markowitz pivot strategy should search maxcol
    // columns and no rows.
    parmlu_[6] = 0.3;

    // the density at which the Markowitz strategy should search only 1 column,
    // or (if storage is available) the density at which all remaining rows and
    // columns will be processed by a dense LU code.
    parmlu_[7] = 0.5;
  }

  LinSolverDirectLUSOL::~LinSolverDirectLUSOL()
  {
    freeSolverData();
    delete L_;
    delete U_;
    L_ = nullptr;
    U_ = nullptr;
  }

  /// @pre A is in the COO format, is a fully expanded matrix, and contains no duplicates
  int LinSolverDirectLUSOL::setup(matrix::Sparse* A,
                                  matrix::Sparse* /* L */,
                                  matrix::Sparse* /* U */,
                                  index_type*     /* P */,
                                  index_type*     /* Q */,
                                  vector_type*  /* rhs */)
  {
    A_ = A;
    is_factorized_ = false;
    delete L_;
    delete U_;
    L_ = nullptr;
    U_ = nullptr;
    return 0;
  }

  /**
   * At this time, only memory allocation and initialization is done here.
   *
   * @return int - 0 if successful, error code otherwise
   *
   * @note LUSOL does not expose symbolic factorization in its API. It might
   *       be possible refactor lu1fac into separate symbolic and numerical
   *       factorization functions, but for now, we do the both in ::factorize().
   */
  int LinSolverDirectLUSOL::analyze()
  {
    // Brute force solution: If the solver workspace is already allocated, nuke it!
    if (is_solver_data_allocated_) {
      freeSolverData();
      is_solver_data_allocated_ = false;
    }

    nelem_ = A_->getNnz();
    m_ = A_->getNumRows();
    n_ = A_->getNumColumns();

    allocateSolverData();
    is_solver_data_allocated_ = true;

    real_type* a_in = A_->getValues(memory::HOST);
    index_type* indc_in = A_->getRowData(memory::HOST);
    index_type* indr_in = A_->getColData(memory::HOST);

    for (index_type i = 0; i < nelem_; i++) {
      a_[i] = a_in[i];
      indc_[i] = indc_in[i] + 1;
      indr_[i] = indr_in[i] + 1;
    }

    return 0;
  }

  int LinSolverDirectLUSOL::factorize()
  {
    // NOTE: this is probably good enough as far as checking goes
    if (a_ == nullptr || indc_ == nullptr || indr_ == nullptr) {
      out::error() << "LUSOL workspace not allocated!\n";
      return -1;
    }

    index_type inform = 0;

    lu1fac(&m_,
           &n_,
           &nelem_,
           &lena_,
           luparm_,
           parmlu_,
           a_,
           indc_,
           indr_,
           p_,
           q_,
           lenc_,
           lenr_,
           locc_,
           locr_,
           iploc_,
           iqloc_,
           ipinv_,
           iqinv_,
           w_,
           &inform);

    is_factorized_ = true;

    // TODO: consider handling inform = 7

    return inform;
  }

  int LinSolverDirectLUSOL::refactorize()
  {
    out::error() << "LinSolverDirect::refactorize() called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return -1;
  }

  int LinSolverDirectLUSOL::solve(vector_type* rhs, vector_type* x)
  {
    if (rhs->getSize() != m_ || x->getSize() != n_ || !is_factorized_) {
      return -1;
    }

    index_type mode = 5;
    index_type inform = 0;

    lu6sol(&mode,
           &m_,
           &n_,
           rhs->getData(memory::HOST),
           x->getData(memory::HOST),
           &lena_,
           luparm_,
           parmlu_,
           a_,
           indc_,
           indr_,
           p_,
           q_,
           lenc_,
           lenr_,
           locc_,
           locr_,
           &inform);

    return inform;
  }

  int LinSolverDirectLUSOL::solve(vector_type* /* x */)
  {
    out::error() << "LinSolverDirect::solve(vector_type*) called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return -1;
  }

  /**
   * @pre The input matrix has been factorized
   *
   * @post A pointer to the L factor of the input matrix is returned in CSC
   *       format. The linear solver instance owns this data
   */
  matrix::Sparse* LinSolverDirectLUSOL::getLFactor()
  {
    if (!is_factorized_) {
      return nullptr;
    }

    if (L_ != nullptr) {
      // because of the way we've implemented setup, we can just return the
      // existing pointer in L_ as this means we've already extracted L
      //
      // this isn't perfect, but it's functional
      return L_;
    }

    index_type diagonal_bound = std::min({m_, n_});
    index_type current_nnz = luparm_[22];

    L_ = static_cast<matrix::Sparse*>(new matrix::Csc(n_, m_, current_nnz + diagonal_bound, false, true));
    L_->allocateMatrixData(memory::HOST);

    index_type* columns = L_->getColData(memory::HOST);
    index_type* rows = L_->getRowData(memory::HOST);
    real_type* values = L_->getValues(memory::HOST);

    // build an inverse permutation array for p

    // NOTE: this is not one-indexed like the original is
    std::unique_ptr<index_type[]> pt = std::unique_ptr<index_type[]>(new index_type[m_]);
    for (index_type i = 0; i < m_; i++) {
      pt[p_[i] - 1] = i;
    }

    // preprocessing since columns are stored unordered within lusol's workspace

    columns[0] = 0;
    index_type offset = lena_ - 1;
    index_type initial_m = luparm_[19];
    for (index_type i = 0; i < initial_m; i++) {
      index_type column_nnz = lenc_[i];
      index_type column_nnz_end = offset - column_nnz;
      index_type corresponding_column = pt[indr_[column_nnz_end + 1] - 1];

      columns[corresponding_column + 1] = column_nnz;
      offset = column_nnz_end;
    }

    for (index_type column = 0; column < m_; column++) {
      columns[column + 1] += columns[column];
    }

    // handle rectangular l factors correctly
    for (index_type column = 0; column < diagonal_bound; column++) {
      columns[column + 1] += column + 1;
      rows[columns[column + 1] - 1] = column;
      values[columns[column + 1] - 1] = 1.0;
    }

    // fill the destination arrays. iterates over the stored columns, depermuting the
    // column indices to fully compute P*L*Pt while sorting each column's contents using
    // insertion sort (where L is the L factor as stored in LUSOL's workspace)

    offset = lena_ - 1;
    for (index_type i = 0; i < initial_m; i++) {
      index_type corresponding_column = pt[indr_[offset - lenc_[i] + 1] - 1];

      for (index_type destination_offset = columns[corresponding_column];
           destination_offset < columns[corresponding_column + 1] - 1;
           destination_offset++) {
        index_type row = pt[indc_[offset] - 1];

        // closest position to the target row
        index_type* closest_position =
            std::lower_bound(&rows[columns[corresponding_column]], &rows[destination_offset], row);

        // destination offset for the element being inserted
        index_type insertion_offset = static_cast<index_type>(closest_position - rows);

        // LUSOL is not supposed to create duplicates. Report error if it does.
        if (rows[insertion_offset] == row && closest_position != &rows[destination_offset]) {
          out::error() << "duplicate element found during LUSOL L factor extraction\n";
          return nullptr;
        }

        for (index_type swap_offset = destination_offset;
             swap_offset > insertion_offset;
             swap_offset--) {
          std::swap(rows[swap_offset], rows[swap_offset - 1]);
          std::swap(values[swap_offset], values[swap_offset - 1]);
        }

        rows[insertion_offset] = row;
        values[insertion_offset] = -a_[offset];

        offset--;
      }
    }

    return L_;
  }

  /**
   * @pre The input matrix has been factorized
   *
   * @post A pointer to the U factor of the input matrix is returned in CSR
   *      format. The linear solver instance owns this data
   */
  matrix::Sparse* LinSolverDirectLUSOL::getUFactor()
  {
    if (!is_factorized_) {
      return nullptr;
    }

    if (U_ != nullptr) {
      // likewise
      return U_;
    }

    index_type current_nnz = luparm_[23];
    index_type n_singularities = luparm_[10];
    U_ = static_cast<matrix::Sparse*>(new matrix::Csr(n_, m_, current_nnz - n_singularities, false, true));
    U_->allocateMatrixData(memory::HOST);

    index_type* rows = U_->getRowData(memory::HOST);
    index_type* columns = U_->getColData(memory::HOST);
    real_type* values = U_->getValues(memory::HOST);

    // build an inverse permutation array for q

    // NOTE: this is not one-indexed like the original is
    std::unique_ptr<index_type[]> qt = std::unique_ptr<index_type[]>(new index_type[n_]);
    for (index_type i = 0; i < n_; i++) {
      qt[q_[i] - 1] = i;
    }

    // preprocessing since rows technically aren't ordered either

    index_type stored_rows = luparm_[15];
    for (index_type stored_row = 0; stored_row < stored_rows; stored_row++) {
      index_type corresponding_row = p_[stored_row] - 1;
      rows[stored_row + 1] = lenr_[corresponding_row];
    }

    for (index_type row = 0; row < n_; row++) {
      rows[row + 1] += rows[row];
    }

    // fill the destination arrays

    for (index_type row = 0; row < n_; row++) {
      index_type offset = locr_[p_[row] - 1] - 1;

      for (index_type destination_offset = rows[row]; destination_offset < rows[row + 1]; destination_offset++) {
        index_type column = qt[indr_[offset] - 1];

        // closest position to the target column
        index_type* closest_position =
            std::lower_bound(&columns[rows[row]], &columns[destination_offset], column);

        // destination offset for the element being inserted
        index_type insertion_offset = static_cast<index_type>(closest_position - columns);

        // LUSOL is not supposed to create duplicates. Report error if it does.
        if (columns[insertion_offset] == column && closest_position != &columns[destination_offset]) {
          out::error() << "duplicate element found during LUSOL U factor extraction\n";
          return nullptr;
        }

        for (index_type swap_offset = destination_offset; swap_offset > insertion_offset; swap_offset--) {
          std::swap(columns[swap_offset], columns[swap_offset - 1]);
          std::swap(values[swap_offset], values[swap_offset - 1]);
        }

        columns[insertion_offset] = column;
        values[insertion_offset] = a_[offset];

        offset++;
      }
    }

    return U_;
  }

  index_type* LinSolverDirectLUSOL::getPOrdering()
  {
    if (P_ == nullptr) {
      P_ = new index_type[m_];
    }

    for (index_type i = 0; i < m_; i++) {
      P_[i] = p_[i] - 1;
    }

    return P_;
  }

  index_type* LinSolverDirectLUSOL::getQOrdering()
  {
    if (Q_ == nullptr) {
      Q_ = new index_type[n_];
    }

    for (index_type i = 0; i < n_; i++) {
      Q_[i] = q_[i] - 1;
    }

    return Q_;
  }

  void LinSolverDirectLUSOL::setPivotThreshold(real_type /* _ */)
  {
    out::error() << "LinSolverDirect::setPivotThreshold(real_type) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  void LinSolverDirectLUSOL::setOrdering(int /* _ */)
  {
    out::error() << "LinSolverDirect::setOrdering(int) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  void LinSolverDirectLUSOL::setHaltIfSingular(bool /* _ */)
  {
    out::error() << "LinSolverDirect::setHaltIfSingular(bool) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  real_type LinSolverDirectLUSOL::getMatrixConditionNumber()
  {
    out::error() << "LinSolverDirect::getMatrixConditionNumber() called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return 0;
  }

  //
  // Private Methods
  //

  int LinSolverDirectLUSOL::allocateSolverData()
  {
    // LUSOL does not do symbolic analysis to determine workspace size to store
    // L and U factors, so we have to guess something. See documentation for 
    // lena_ in resolve/lusol/lusol.f90 file.
    lena_ = std::max({20 * nelem_, 10 * m_, 10 * n_, 10000});

    a_ = new real_type[lena_];
    indc_ = new index_type[lena_];
    indr_ = new index_type[lena_];
    mem_.setZeroArrayOnHost(a_, lena_);
    mem_.setZeroArrayOnHost(indc_, lena_);
    mem_.setZeroArrayOnHost(indr_, lena_);

    p_ = new index_type[m_];
    mem_.setZeroArrayOnHost(p_, m_);

    q_ = new index_type[n_];
    mem_.setZeroArrayOnHost(q_, n_);

    lenc_ = new index_type[n_];
    mem_.setZeroArrayOnHost(lenc_, n_);

    lenr_ = new index_type[m_];
    mem_.setZeroArrayOnHost(lenr_, m_);

    locc_ = new index_type[n_];
    mem_.setZeroArrayOnHost(locc_, n_);

    locr_ = new index_type[m_];
    mem_.setZeroArrayOnHost(locr_, m_);

    iploc_ = new index_type[n_];
    mem_.setZeroArrayOnHost(iploc_, n_);

    iqloc_ = new index_type[m_];
    mem_.setZeroArrayOnHost(iqloc_, m_);

    ipinv_ = new index_type[m_];
    mem_.setZeroArrayOnHost(ipinv_, m_);

    iqinv_ = new index_type[n_];
    mem_.setZeroArrayOnHost(iqinv_, n_);

    w_ = new real_type[n_];
    mem_.setZeroArrayOnHost(w_, n_);

    return 0;
  }

  int LinSolverDirectLUSOL::freeSolverData()
  {
    delete[] a_;
    delete[] indc_;
    delete[] indr_;
    delete[] p_;
    delete[] q_;
    delete[] lenc_;
    delete[] lenr_;
    delete[] locc_;
    delete[] locr_;
    delete[] iploc_;
    delete[] iqloc_;
    delete[] ipinv_;
    delete[] iqinv_;
    delete[] w_;
    delete[] P_;
    delete[] Q_;
    a_ = nullptr;
    indc_ = nullptr;
    indr_ = nullptr;
    p_ = nullptr;
    q_ = nullptr;
    lenc_ = nullptr;
    lenr_ = nullptr;
    locc_ = nullptr;
    locr_ = nullptr;
    iploc_ = nullptr;
    iqloc_ = nullptr;
    ipinv_ = nullptr;
    iqinv_ = nullptr;
    w_ = nullptr;
    P_ = nullptr;
    Q_ = nullptr;

    return 0;
  }
} // namespace ReSolve
