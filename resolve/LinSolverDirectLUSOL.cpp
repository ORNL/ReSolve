#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>

#include "LinSolverDirectLUSOL.hpp"
#include "lusol/lusol.hpp"

#include <resolve/matrix/Csc.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using log = io::Logger;

  LinSolverDirectLUSOL::LinSolverDirectLUSOL()
  {
    luparm_[0] = 6;

    switch (log::verbosity()) {
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
    delete L_;
    delete U_;
    delete[] P_;
    delete[] Q_;
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
    L_ = U_ = nullptr;
    P_ = Q_ = indc_ = indr_ = p_ = q_ = lenc_ = lenr_ = locc_ = locr_ = iploc_ =
        iqloc_ = ipinv_ = iqinv_ = nullptr;
    a_ = w_ = nullptr;
  }

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

  int LinSolverDirectLUSOL::analyze()
  {
    // TODO: replace this with something better
    if (a_ != nullptr || indc_ != nullptr || indr_ != nullptr) {
        return -1;
    }

    // NOTE: LUSOL does not come with any discrete analysis operation. it is
    //       possible to break apart bits of lu1fac into that, but for now,
    //       we don't bother and shunt it all into ::factorize()

    nelem_ = A_->getNnz();
    m_ = A_->getNumRows();
    n_ = A_->getNumColumns();

    // NOTE: determines a hopefully "good enough" size for a_, indc_, indr_.
    //       see lena_'s documentation for more details
    if (nelem_ >= parmlu_[7] * m_ * n_) {
      lena_ = m_ * n_;
    } else {
      lena_ = std::min(5 * nelem_, 2 * m_ * n_);
    }

    // NOTE: this is extremely unsafe. this should be removed the moment
    //       a safer alternative is available

    // INVARIANT: the input matrix is of the coo format

    real_type* a_in = A_->getValues(memory::HOST);
    index_type* indc_in = A_->getRowData(memory::HOST);
    index_type* indr_in = A_->getColData(memory::HOST);

    a_ = new real_type[lena_];
    indc_ = new index_type[lena_];
    indr_ = new index_type[lena_];

    for (index_type i = 0; i < nelem_; i++) {
      a_[i] = a_in[i];
      indc_[i] = indc_in[i] + 1;
      indr_[i] = indr_in[i] + 1;
    }

    p_ = new index_type[m_];
    std::fill_n(p_, m_, 0);

    q_ = new index_type[n_];
    std::fill_n(q_, n_, 0);

    lenc_ = new index_type[n_];
    std::fill_n(lenc_, n_, 0);

    lenr_ = new index_type[m_];
    std::fill_n(lenr_, m_, 0);

    locc_ = new index_type[n_];
    std::fill_n(locc_, n_, 0);

    locr_ = new index_type[m_];
    std::fill_n(locr_, m_, 0);

    iploc_ = new index_type[n_];
    std::fill_n(iploc_, n_, 0);

    iqloc_ = new index_type[m_];
    std::fill_n(iqloc_, m_, 0);

    ipinv_ = new index_type[m_];
    std::fill_n(ipinv_, m_, 0);

    iqinv_ = new index_type[n_];
    std::fill_n(iqinv_, n_, 0);

    w_ = new real_type[n_];
    std::fill_n(w_, n_, 0);

    return 0;
  }

  int LinSolverDirectLUSOL::factorize()
  {
    // NOTE: this is probably good enough as far as checking goes
    if (a_ == nullptr || indc_ == nullptr || indr_ == nullptr) {
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
    log::error() << "LinSolverDirect::refactorize() called on "
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
    log::error() << "LinSolverDirect::solve(vector_type*) called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return -1;
  }

  matrix::Sparse* LinSolverDirectLUSOL::getLFactor()
  {
    log::error() << "LinSolverDirect::getLFactor() called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return nullptr;
  }

  matrix::Sparse* LinSolverDirectLUSOL::getUFactor()
  {
    log::error() << "LinSolverDirect::getUFactor() called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return nullptr;
  }

  index_type* LinSolverDirectLUSOL::getPOrdering()
  {
    log::error() << "LinSolverDirect::getPOrdering() called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return nullptr;
  }

  index_type* LinSolverDirectLUSOL::getQOrdering()
  {
    log::error() << "LinSolverDirect::getQOrdering() called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return nullptr;
  }

  void LinSolverDirectLUSOL::setPivotThreshold(real_type /* _ */)
  {
    log::error() << "LinSolverDirect::setPivotThreshold(real_type) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  void LinSolverDirectLUSOL::setOrdering(int /* _ */)
  {
    log::error() << "LinSolverDirect::setOrdering(int) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  void LinSolverDirectLUSOL::setHaltIfSingular(bool /* _ */)
  {
    log::error() << "LinSolverDirect::setHaltIfSingular(bool) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  real_type LinSolverDirectLUSOL::getMatrixConditionNumber()
  {
    log::error() << "LinSolverDirect::getMatrixConditionNumber() called on "
                    "LinSolverDirectLUSOL which is unimplemented!\n";
    return 0;
  }
} // namespace ReSolve
