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
    luparm_[0] = 6;

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

    // TODO: these are defaults. make these configurable in the future using the
    //       forthcoming parameter manager

    luparm_[2] = 5;
    luparm_[5] = 0;
    luparm_[7] = 1;

    // NOTE: the default value of this is suggested to change based upon the
    //       pivoting strategy in use. see the LUSOL source for more information
    parmlu_[0] = 10.0;

    parmlu_[1] = 10.0;

    // NOTE: there's ReSolve::constants::DEFAULT_TOL but this carries with it
    //       the implication that it's configurable. i'm unaware of any such
    //       location to configure it and it's likely related to the parameter
    //       manager mentioned above
    //
    //       so, for now, we use the suggested default in LUSOL
    parmlu_[2] = std::pow(std::numeric_limits<real_type>::epsilon(), 0.8);

    // TODO: figure out where this exponent comes from :)
    parmlu_[4] = parmlu_[3] = std::pow(std::numeric_limits<real_type>::epsilon(), 0.67);

    parmlu_[5] = 3.0;
    parmlu_[6] = 0.3;
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
                                  index_type* /* P */,
                                  index_type* /* Q */,
                                  vector_type* /* rhs */)
  {
    A_ = A;
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
    if (rhs->getSize() != m_ || x->getSize() != n_) {
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

  matrix::Sparse* LinSolverDirectLUSOL::getLFactor()
  {
    matrix::Csc* L = new matrix::Csc(n_, m_, luparm_[22] + std::min({m_, n_}), false, true);
    L->allocateMatrixData(memory::HOST);

    index_type* columns = L->getColData(memory::HOST);
    index_type* rows = L->getRowData(memory::HOST);
    real_type* values = L->getValues(memory::HOST);

    // build an inverse permutation array for p

    // NOTE: this is not one-indexed like the original is
    std::unique_ptr<index_type[]> pt = std::unique_ptr<index_type[]>(new index_type[m_]);
    for (index_type i = 0; i < m_; i++) {
      pt[p_[i] - 1] = i;
    }

    // preprocessing since columns are stored unordered within lusol's workspace

    columns[0] = 0;
    index_type offset = lena_ - 1;
    for (index_type i = 0; i < luparm_[19]; i++) {
      index_type column_nnz = lenc_[i];
      index_type column_nnz_end = offset - column_nnz;
      index_type corresponding_column = pt[indr_[column_nnz_end + 1] - 1];

      columns[corresponding_column + 1] = column_nnz;
      offset = column_nnz_end;
    }

    for (index_type column = 0; column < m_; column++) {
      columns[column + 1] += columns[column];
    }

    // fill the destination arrays
    // TODO: sort the destination arrays by row
    // TODO: add ones along the diagonal
    // TODO: use the already allocated L_ and U_ matrices instead of allocating new ones
    // TODO: size appears to be constrained by nsing

    offset = lena_ - 1;
    for (index_type i = 0; i < luparm_[19]; i++) {
      index_type corresponding_column = pt[indr_[offset - lenc_[i] + 1] - 1];

      for (index_type destination_offset = columns[corresponding_column];
           destination_offset < columns[corresponding_column + 1];
           destination_offset++) {
        rows[destination_offset] = pt[indc_[offset] - 1];
        values[destination_offset] = a_[offset];
        offset--;
      }
    }

    return static_cast<matrix::Sparse*>(L);
  }

  matrix::Sparse* LinSolverDirectLUSOL::getUFactor()
  {
    matrix::Csr* U = new matrix::Csr(n_, m_, luparm_[23], false, true);
    U->allocateMatrixData(memory::HOST);

    index_type* rows = U->getRowData(memory::HOST);
    index_type* columns = U->getColData(memory::HOST);
    real_type* values = U->getValues(memory::HOST);

    // build an inverse permutation array for q

    // NOTE: this is not one-indexed like the original is
    std::unique_ptr<index_type[]> qt = std::unique_ptr<index_type[]>(new index_type[n_]);
    for (index_type i = 0; i < n_; i++) {
      qt[q_[i] - 1] = i;
    }

    // preprocessing since rows technically aren't ordered either

    for (index_type stored_row = 0; stored_row < luparm_[15]; stored_row++) {
      index_type corresponding_row = p_[stored_row] - 1;
      rows[stored_row + 1] = lenr_[corresponding_row];
    }

    for (index_type row = 0; row < n_; row++) {
      rows[row + 1] += rows[row];
    }

    // fill the destination arrays

    for (index_type row = 0; row < n_; row++) {
      index_type offset = locr_[p_[row] - 1] - 1;

      for (index_type destination_offset = rows[row];
           destination_offset < rows[row + 1];
           destination_offset++) {
        columns[destination_offset] = qt[indr_[offset] - 1];
        values[destination_offset] = a_[offset];
        offset++;
      }
    }

    return static_cast<matrix::Sparse*>(U);
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
    // NOTE: determines a hopefully "good enough" size for a_, indc_, indr_.
    //       see lena_'s documentation for more details
    lena_ = std::max({2 * nelem_, 10 * m_, 10 * n_, 10000});

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
