#include <cmath>
#include <cstdlib>
#include <limits>

#include "LinSolverDirectLUSOL.hpp"
#include "lusol/lusol.hpp"

#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Utilities.hpp>
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
    parmlu_[4] = parmlu_[3] =
        std::pow(std::numeric_limits<real_type>::epsilon(), 0.67);

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
    free(a_);
    free(indc_);
    free(indr_);
    free(p_);
    free(q_);
    free(lenc_);
    free(lenr_);
    free(locc_);
    free(locr_);
    free(iploc_);
    free(iqloc_);
    free(ipinv_);
    free(iqinv_);
    free(w_);
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
    if (matrix::expand(A_) != 0) {
      return -1;
    }
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

    a_ = static_cast<real_type*>(std::realloc(a_, lena_ * sizeof(real_type)));
    indc_ = static_cast<index_type*>(
        std::realloc(indc_, lena_ * sizeof(index_type)));
    indr_ = static_cast<index_type*>(
        std::realloc(indr_, lena_ * sizeof(index_type)));

    auto f = matrix::elements(A_);
    bool ok;
    std::tuple<index_type, index_type, real_type> t;
    index_type i = 0, j, k;
    real_type x;

    for (std::tie(t, ok) = f(); ok; std::tie(t, ok) = f()) {
      std::tie(j, k, x) = t;

      indc_[i] = j + 1;
      indr_[i] = k + 1;
      a_[i] = x;

      i++;
    }

    p_ = static_cast<index_type*>(std::realloc(p_, m_ * sizeof(index_type)));
    std::fill_n(p_, m_, 0);

    q_ = static_cast<index_type*>(std::realloc(q_, n_ * sizeof(index_type)));
    std::fill_n(q_, n_, 0);

    lenc_ = static_cast<index_type*>(std::realloc(lenc_, n_ * sizeof(index_type)));
    std::fill_n(lenc_, n_, 0);

    lenr_ = static_cast<index_type*>(std::realloc(lenr_, m_ * sizeof(index_type)));
    std::fill_n(lenr_, m_, 0);

    locc_ = static_cast<index_type*>(std::realloc(locc_, n_ * sizeof(index_type)));
    std::fill_n(locc_, n_, 0);

    locr_ = static_cast<index_type*>(std::realloc(locr_, m_ * sizeof(index_type)));
    std::fill_n(locr_, m_, 0);

    iploc_ = static_cast<index_type*>(std::realloc(iploc_, n_ * sizeof(index_type)));
    std::fill_n(iploc_, n_, 0);

    iqloc_ = static_cast<index_type*>(std::realloc(iqloc_, m_ * sizeof(index_type)));
    std::fill_n(iqloc_, m_, 0);

    ipinv_ = static_cast<index_type*>(std::realloc(ipinv_, m_ * sizeof(index_type)));
    std::fill_n(ipinv_, m_, 0);

    iqinv_ = static_cast<index_type*>(std::realloc(iqinv_, n_ * sizeof(index_type)));
    std::fill_n(iqinv_, n_, 0);

    w_ = static_cast<real_type*>(std::realloc(w_, n_ * sizeof(real_type)));
    std::fill_n(w_, n_, 0);

    return 0;
  }

  int LinSolverDirectLUSOL::analyze()
  {
    // NOTE: LUSOL does not come with any discrete analysis operation. it is
    //       possible to break apart bits of lu1fac into that, but for now,
    //       we don't bother and shunt it all into ::factorize()
    return 0;
  }

  int LinSolverDirectLUSOL::factorize()
  {
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

    // NOTE: this is probably enough to be handled correctly by most callees
    return -inform;
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

    // NOTE: ditto above
    return -inform;
  }

  int LinSolverDirectLUSOL::solve(vector_type* x)
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

  void LinSolverDirectLUSOL::setPivotThreshold(real_type _)
  {
    log::error() << "LinSolverDirect::setPivotThreshold(real_type) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  void LinSolverDirectLUSOL::setOrdering(int _)
  {
    log::error() << "LinSolverDirect::setOrdering(int) called on "
                    "LinSolverDirectLUSOL on which it is irrelevant!\n";
  }

  void LinSolverDirectLUSOL::setHaltIfSingular(bool _)
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
