/**
 * @file HyKKTSolver.cpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief HyKKT system solver implementation
 */

#include "HyKKTSolver.hpp"

#include <resolve/matrix/io.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve
{
  using namespace constants;
  using out = io::Logger;

  /**
   * @brief basic constructor
   *
   * @param[in] n_x   - Number of rows of the 1st row of blocks of the system
   * @param[in] m_d   - Number of rows of the 2nd row of blocks of the system
   * @param[in] m_c   - Number of rows of the 3rd row of blocks of the system
   */
  hykkt::HyKKTSolver::HyKKTSolver(index_type n_x, index_type m_d, index_type m_c, memory::MemorySpace memspace)
    : n_x_{n_x},
      m_d_{m_d},
      m_c_{m_c},
      n_total_{n_x + m_c},
      memspace_{memspace}
  {
  }

  hykkt::HyKKTSolver::~HyKKTSolver()
  {
    delete r_x_perm_;
    delete omega_perm_;
    delete schur_;
    delete D_s_vals_;
    delete r_yd_scaled_;
    delete r_x_til_;
    delete r_x_hat_;
    delete z_;
    delete r_y_copy_;
    delete J_tr_;
    delete J_d_tr_;
    delete H_gamma_perm_;
    delete J_perm_;
    delete J_tr_perm_;
    delete J_d_scaled_;
    delete H_tilde_;
    delete H_gamma_;
    delete J_copy_;
    delete J_tr_copy_;
    delete ruiz_;
    delete spgemm_htil_;
    delete spgemm_hgamma_;
    delete permutation_;
    delete cholesky_;
    delete sccg_;
  }

  /**
   * @brief Sets blocks of the KKT matrix in CSR format to user provided
   * values. It will only set pointers to user provided data; it is user's
   * responsibility to supply and later delete that memory.
   *
   * Reusing the solver requires the sparsity patterns of the matrix blocks
   * to remain unchanged. Changing J_d between empty and nonempty is rejected.
   *
   * @param[in] H_plus_D_x - Pointer to the Hessian matrix block (n_x x n_x),
   * corresponding to H + D_x in the HyKKT paper.
   * @param[in] D_s - Pointer to the slack variables derivatives matrix block
   * (m_d x m_d).
   * @param[in] J - Pointer to the equality constraints Jacobian block
   * (m_c x n_x).
   * @param[in] J_d - Pointer to the inequality constraints Jacobian block
   * (m_d x n_x)
   *
   * @return 0 if successful, 1 if the supplied blocks are incompatible with
   *         the allocated solver state. On failure, the previously stored
   *         blocks are left unchanged.
   */
  int hykkt::HyKKTSolver::setMatrixBlocks(matrix::Csr* H_plus_D_x, matrix::Csr* D_s, matrix::Csr* J, matrix::Csr* J_d)
  {
    const bool J_d_flag = J_d->getNnz() > 0;

    if (allocated_ && J_d_flag_ != J_d_flag)
    {
      out::error() << "Changing J_d between empty and nonempty is not "
                      "supported when reusing HyKKT.\n";
      return 1;
    }

    H_   = H_plus_D_x;
    D_s_ = D_s;
    J_   = J;
    J_d_ = J_d;

    if (!allocated_)
    {
      J_d_flag_ = J_d_flag;
    }
    return 0;
  }

  /**
   * @brief Sets the blocks of the RHS vector of the system to user provided
   * values. It will only set pointers to user provided data; it is user's
   * responsibility to supply and later delete that memory.
   *
   * @param[in] r_x - Pointer to the r_x vector (shape: n_x)
   * @param[in] r_s - Pointer to the r_s vector (shape: m_d)
   * @param[in] r_y - Pointer to the r_y vector (shape: m_c)
   * @param[in] r_yd - Pointer to the r_yd vector (shape: m_d)
   */
  void hykkt::HyKKTSolver::setRHSBlocks(vector::Vector* r_x, vector::Vector* r_s, vector::Vector* r_y, vector::Vector* r_yd)
  {
    r_x_  = r_x;
    r_s_  = r_s;
    r_y_  = r_y;
    r_yd_ = r_yd;
  }

  /**
   * @brief Sets the pointers to the blocks of the LHS (output) vector of the
   * system. It will set pointers to the vector object that will contain the
   * solver's solutions. Existing values will not be used and will be overridden.
   * It is user's responsibility to supply and later delete that memory.
   *
   * @param[in] x - Pointer to the x vector (shape: n_x)
   * @param[in] s - Pointer to the s vector (shape: m_d)
   * @param[in] y - Pointer to the y vector (shape: m_c)
   * @param[in] y_d - Pointer to the y_d vector (shape: m_d)
   */
  void hykkt::HyKKTSolver::setLHSPointers(vector::Vector* x, vector::Vector* s, vector::Vector* y, vector::Vector* y_d)
  {
    x_   = x;
    s_   = s;
    y_   = y;
    y_d_ = y_d;
  }

  /*
   * @brief sets gamma value for hykkt solver
   *
   * @param[in] gamma - new value for gamma_
   *
   * @post gamma_ is now equal to gamma
   */
  void hykkt::HyKKTSolver::setGamma(real_type gamma)
  {
    gamma_ = gamma;
  }

  /**
   * @brief Loads or reloads pointer to the matrix and vector handlers
   * for matrix and vector operations.
   * @param[in] matrixHandler - New matrix handler
   * @param[in] vectorHandler - New vector handler
   */
  void hykkt::HyKKTSolver::addHandlers(MatrixHandler* matrixHandler, VectorHandler* vectorHandler)
  {
    matrixHandler_ = matrixHandler;
    vectorHandler_ = vectorHandler;
  }

  /*
   * @brief uses Hykkt algorithm to solve KKT system
   *
   * @pre matrix files have been loaded into the solver
   *
   * @param[out] - Error of Ax - b
   *
   * @post solution to given KKT system is computed using Hykkt
   */
  real_type hykkt::HyKKTSolver::solve()
  {
    setupParameters();

    if (!allocated_)
    {
      setupSpGEMMHtilde();
    }
    computeSpGEMMHtilde();

    setupSolutionCheck();

    if (!allocated_)
    {
      setupRuizScaling();
    }
    computeRuizScaling();

    if (!allocated_)
    {
      setupSpGEMMHgamma();
    }
    computeSpGEMMHgamma();

    if (!allocated_)
    {
      setupPermutation();
    }
    applyPermutation();

    if (!allocated_)
    {
      setupHgammaFactorization();
    }
    computeHgammaFactorization();

    if (!allocated_)
    {
      sccg_ = new SchurComplementConjugateGradient(J_->getNumRows(),
                                                   J_->getNumColumns(),
                                                   cholesky_,
                                                   matrixHandler_,
                                                   vectorHandler_,
                                                   memspace_);
      sccg_->setup();
    }
    setupConjugateGradient();
    computeConjugateGradient();

    recoverSolution();
    return checkError();
  }

  /**
   * @brief allocates and initiates variables for KKT system
   *
   * @pre J_d_flag_ determines whether variables used to form H_tilde with
   *      SpGEMM should be initialized.
   *
   * @post all variables used for hykkt are allocated for; J_d-
   *       related variables are not initiated if J_d nnz == 0
   */
  void hykkt::HyKKTSolver::setupParameters()
  {
    // Assume all matrix blocks and RHS blocks are already set

    std::cout << "H size: " << H_->getNumRows() << " " << H_->getNumColumns() << " " << H_->getNnz() << " \n";
    std::cout << "J size: " << J_->getNumRows() << "  " << J_->getNumColumns() << "  " << J_->getNnz() << " \n";
    std::cout << "D_s nnz = " << D_s_->getNnz() << "\n";

    if (!allocated_)
    {
      r_x_perm_    = new vector::Vector(n_x_);
      omega_perm_  = new vector::Vector(n_x_);
      schur_       = new vector::Vector(m_c_);
      D_s_vals_    = new vector::Vector(m_d_);
      r_yd_scaled_ = new vector::Vector(r_yd_->getSize());
      r_x_til_     = new vector::Vector(n_x_);
      r_x_hat_     = new vector::Vector(n_x_);
      z_           = new vector::Vector(n_x_);
      r_y_copy_    = new vector::Vector(m_c_);
      J_tr_        = new matrix::Csr(J_->getNumColumns(), J_->getNumRows(), J_->getNnz());
      J_d_tr_      = new matrix::Csr(J_d_->getNumColumns(), J_d_->getNumRows(), J_d_->getNnz());
      H_tilde_     = new matrix::Csr(H_->getNumRows(), H_->getNumColumns(), H_->getNnz());
      J_perm_      = new matrix::Csr(J_->getNumRows(), J_->getNumColumns(), J_->getNnz());
      J_tr_perm_   = new matrix::Csr(J_tr_->getNumRows(), J_tr_->getNumColumns(), J_tr_->getNnz());
      J_d_scaled_  = new matrix::Csr(J_d_->getNumRows(), J_d_->getNumColumns(), J_d_->getNnz());

      r_yd_scaled_->allocate(memspace_);
      r_x_perm_->allocate(memspace_);
      omega_perm_->allocate(memspace_);
      schur_->allocate(memspace_);
      r_x_til_->allocate(memspace_);
      r_x_hat_->allocate(memspace_);
      z_->allocate(memspace_);
      r_y_copy_->allocate(memspace_);
      J_tr_->allocateAll(memspace_);
      J_d_tr_->allocateMatrixData(memspace_);
      J_perm_->allocateAll(memspace_);
      J_tr_perm_->allocateAll(memspace_);
      J_d_scaled_->allocateWithExternalSparsityPattern(J_d_->getRowData(memspace_), J_d_->getColData(memspace_), J_d_->getNnz(), memspace_);
      // H_tilde_ does not need to be allocated because loadResultMatrix() does it later
    }

    // D_s may be replaced between solves, so refresh the external value pointer.
    D_s_vals_->setData(D_s_->getValues(memspace_), memspace_);
    r_y_copy_->copyFromExternal(r_y_, memspace_, memspace_);

    // Matrix values may change between solves, so refresh the transpose.
    matrixHandler_->transpose(J_, J_tr_, memspace_);
    if (J_d_flag_)
    {
      matrixHandler_->transpose(J_d_, J_d_tr_, memspace_);
      J_d_scaled_->copyValues(J_d_->getValues(memspace_), memspace_, memspace_);
    }
  }

  /**
   * @brief Creates the SpGEMM solver for the H_tildeda matrix and
   * loads the result matrix pointer
   * @post The result of the solver is set to be written into H_tilde_
   */
  void hykkt::HyKKTSolver::setupSpGEMMHtilde()
  {
    spgemm_htil_ = new SpGEMM(memspace_, ONE, ONE);
  }

  /*
   * @brief computes SpGEMM to calculate H_tildeda matrix
   *
   * @pre matrices and spgemm_htil_ properly allocated using setup
   *      method for spgemm_htil_
   *
   * @post H_tildeda calculated using J_d matrix if J_d nnz > 0 and
   *       is set to H if J_d nnz == 0
   */
  void hykkt::HyKKTSolver::computeSpGEMMHtilde()
  {
    if (J_d_flag_)
    {
      matrixHandler_->leftScale(D_s_vals_, J_d_scaled_, memspace_);
      spgemm_htil_->loadProductMatrices(J_d_tr_, J_d_scaled_);
      spgemm_htil_->loadSumMatrix(H_);
      spgemm_htil_->loadResultMatrix(&H_tilde_); // H_tilde_ will be created by SpGEMM at this step

      r_yd_scaled_->copyFromExternal(r_yd_, memspace_, memspace_);
      vectorHandler_->scal(D_s_vals_, r_yd_scaled_, memspace_);

      // r_yd_scaled_ = r_s + D_s * r_yd
      vectorHandler_->axpy(ONE, r_s_, r_yd_scaled_, memspace_);
      r_x_til_->copyFromExternal(r_x_, memspace_, memspace_);
      matrixHandler_->matvec(J_d_tr_, r_yd_scaled_, r_x_til_, &ONE, &ONE, memspace_);
      spgemm_htil_->compute();
    }
    else
    {
      H_tilde_->setNnz(H_->getNnz());
      if (!allocated_)
      {
        H_tilde_->allocateMatrixData(memspace_);
      }
      H_tilde_->copyFromExternal(H_->getRowData(memspace_),
                                 H_->getColData(memspace_),
                                 H_->getValues(memspace_),
                                 memspace_,
                                 memspace_);

      r_x_til_->copyFromExternal(r_x_, memspace_, memspace_);
    }
  }

  /*
   * @brief Copies the matrices J and J^T which are later overwritten so
   *        the solution can be checked
   *
   * @pre Matrices J and J^T are properly allocated and initialized.
   *
   * @post J and J^T are copied
   *
   */
  void hykkt::HyKKTSolver::setupSolutionCheck()
  {
    if (!allocated_)
    {
      J_copy_ = new matrix::Csr(J_->getNumRows(), J_->getNumColumns(), J_->getNnz());
      J_copy_->allocateMatrixData(memspace_);
      J_tr_copy_ = new matrix::Csr(J_tr_->getNumRows(), J_tr_->getNumColumns(), J_tr_->getNnz());
      J_tr_copy_->allocateMatrixData(memspace_);
    }
    J_copy_->copyFromExternal(J_->getRowData(memspace_),
                              J_->getColData(memspace_),
                              J_->getValues(memspace_),
                              memspace_,
                              memspace_);
    J_tr_copy_->copyFromExternal(J_tr_->getRowData(memspace_),
                                 J_tr_->getColData(memspace_),
                                 J_tr_->getValues(memspace_),
                                 memspace_,
                                 memspace_);
  }

  /**
   * @brief Creates the Ruiz scaling class
   */
  void hykkt::HyKKTSolver::setupRuizScaling()
  {
    ruiz_ = new RuizScaling(n_x_, n_total_, memspace_);
  }

  /*
   * @brief computes Ruiz scaling so we can judge the size of
   *        Gamma and delta min relative to H Gammma system
   *
   * @pre matrices, RHS, and aggregate scaling vector updated
   * using setup method for ruiz_scaling
   *
   * @post max_d_ now contains the aggregated Ruiz scaling
   */
  void hykkt::HyKKTSolver::computeRuizScaling()
  {
    ruiz_->addMatrixData(H_tilde_, J_, J_tr_);
    ruiz_->addRhsData(r_x_til_, r_y_);
    ruiz_->scale(ruiz_its_);
    max_d_ = ruiz_->getAggregateScalingVector();
  }

  /**
   * @brief Creates the SpGEMM solver for the H_gamma matrix and
   * loads the result matrix pointer
   * @post The result of the solver is set to be written into H_gamma_
   */
  void hykkt::HyKKTSolver::setupSpGEMMHgamma()
  {
    spgemm_hgamma_ = new SpGEMM(memspace_, gamma_, ONE);
  }

  /*
   * @brief computes SpGEMM to calculate H_gamma matrix
   *
   * @pre matrices and spgemm_hgamma properly allocated using setup
   *      method for spgemm_hgamma
   *
   * @post H_gamma CSR now represents J_tr * J + H_tilde
   */
  void hykkt::HyKKTSolver::computeSpGEMMHgamma()
  {
    // Numerical values can change between solves while the sparsity pattern
    // remains fixed, so refresh the SpGEMM inputs before recomputing H_gamma.
    spgemm_hgamma_->setCoefficients(gamma_, ONE);
    spgemm_hgamma_->loadProductMatrices(J_tr_, J_);
    spgemm_hgamma_->loadSumMatrix(H_tilde_);
    // HIP initializes the result descriptor using dimensions established by
    // the product and sum inputs, so load the result matrix after both inputs.
    spgemm_hgamma_->loadResultMatrix(&H_gamma_);
    spgemm_hgamma_->compute();
    r_x_hat_->copyFromExternal(r_x_til_, memspace_, memspace_);
    matrixHandler_->matvec(J_tr_, r_y_, r_x_hat_, &gamma_, &ONE, memspace_);
  }

  void hykkt::HyKKTSolver::setupPermutation()
  {
    H_gamma_perm_ = new matrix::Csr(n_x_, n_x_, H_gamma_->getNnz()); // Nnz not known at setupParameters(), so we have to create it here
    H_gamma_perm_->allocateAll(memspace_);

    if (memspace_ == memory::DEVICE)
    {
      H_gamma_->allocateMatrixData(memory::HOST);
      H_gamma_->syncData(memory::HOST);
      J_tr_->syncData(memory::HOST);
    }

    // These permutation steps are device-only
    permutation_ = new Permutation(n_x_, m_c_, H_gamma_->getNnz(), J_->getNnz(), memspace_);
    permutation_->addMatrixInfo(H_gamma_, J_, J_tr_);
    permutation_->symAmd();
    permutation_->invertPerm();

    permutation_->vecMapRC(H_gamma_perm_->getRowData(memory::HOST), H_gamma_perm_->getColData(memory::HOST));
    H_gamma_perm_->setUpdated(memory::HOST);
    index_type* j_rows      = J_->getRowData(memory::HOST);
    index_type* j_perm_rows = J_perm_->getRowData(memory::HOST);

    for (index_type i = 0; i <= J_->getNumRows(); ++i)
    {
      j_perm_rows[i] = j_rows[i];
    }

    permutation_->vecMapC(J_perm_->getColData(memory::HOST));
    J_perm_->setUpdated(memory::HOST);

    permutation_->vecMapR(J_tr_perm_->getRowData(memory::HOST), J_tr_perm_->getColData(memory::HOST));
    J_tr_perm_->setUpdated(memory::HOST);

    if (memspace_ == memory::DEVICE)
    {
      H_gamma_perm_->syncData(memory::DEVICE);
      J_perm_->syncData(memory::DEVICE);
      J_tr_perm_->syncData(memory::DEVICE);
    }
  }

  /*
  * @brief Applies permutations to values of H_gamma, J, J^T matrices
            and r_x_hat vector
  *
  * @pre Permutation maps for the matrices and vector
  *      computed using setupPermutation()
  *
  *
  * @post H_gamma_perm_, J_perm_, J_tr_perm_, r_x_perm_ now contain permuted
  *       values of H_gamma_, J_, J_tr_, r_x_hat_
  */
  void hykkt::HyKKTSolver::applyPermutation()
  {
    permutation_->mapIndex(PERM_HES_V,
                           H_gamma_->getValues(memspace_),
                           H_gamma_perm_->getValues(memspace_));
    permutation_->mapIndex(PERM_JAC_V,
                           J_->getValues(memspace_),
                           J_perm_->getValues(memspace_));
    permutation_->mapIndex(PERM_JAC_TR_V,
                           J_tr_->getValues(memspace_),
                           J_tr_perm_->getValues(memspace_));
    permutation_->mapIndex(PERM_V,
                           r_x_hat_->getData(memspace_),
                           r_x_perm_->getData(memspace_));
    if (memspace_ == memory::DEVICE)
    {
      H_gamma_perm_->setNotUpdated(memory::HOST);
      H_gamma_perm_->syncData(memory::HOST);
      J_perm_->setNotUpdated(memory::HOST);
      J_tr_perm_->setNotUpdated(memory::HOST);
    }
    r_x_perm_->setDataUpdated(memspace_);
  }

  void hykkt::HyKKTSolver::setupHgammaFactorization()
  {
    cholesky_ = new CholeskySolver(memspace_);
    cholesky_->addMatrixInfo(H_gamma_perm_);
    cholesky_->symbolicAnalysis();
    cholesky_->setPivotTolerance(cholesky_tol_);
  }

  /**
   * @brief sparse Cholesky factorization on permuted (1,1) block
   *        so that LDLt does not have to be used
   *
   * @pre symbolic analysis already computed using setup method
   *      for hgamma_factorization
   *
   * @post H_gamma numerical factorization is computed, thus updating
   *       H_gamma_perm_
   */
  void hykkt::HyKKTSolver::computeHgammaFactorization()
  {
    cholesky_->numericalFactorization();
  }

  void hykkt::HyKKTSolver::setupConjugateGradient()
  {
    cholesky_->solve(omega_perm_, r_x_perm_);
    schur_->copyFromExternal(r_y_, memspace_, memspace_);
    matrixHandler_->matvec(J_perm_, omega_perm_, schur_, &ONE, &MINUS_ONE, memspace_);

    sccg_->addMatrixInfo(J_perm_, J_tr_perm_);
    y_->setToZero(memspace_);
    sccg_->addVectorInfo(y_, schur_);
  }

  /**
   * @brief iterative solver on the Schur complement
   *
   * @pre matrices and sccg_ properly allocated using setup
   *      method for conjugate_gradient
   *
   * @post converged to approximate solution of block system
   */
  void hykkt::HyKKTSolver::computeConjugateGradient()
  {
    sccg_->solve();
  }

  /**
   * @brief recovers solution from hykkt solver
   *
   * @pre execute functions setup and computed correctly
   *
   * @post r_x_, r_s_, r_y_copy_, and r_yd_ contain the solution
   *       on the device
   */
  void hykkt::HyKKTSolver::recoverSolution()
  {
    matrixHandler_->matvec(J_tr_perm_,
                           y_,
                           r_x_perm_,
                           &MINUS_ONE,
                           &ONE,
                           memspace_);
    // block-recovering the solution to the original system by parts
    // this part is to recover delta_x
    cholesky_->solve(z_, r_x_perm_);
    permutation_->mapIndex(REV_PERM_V, z_->getData(memspace_), x_->getData(memspace_));
    x_->setDataUpdated(memspace_);

    // scale back delta_y and delta_x (ever_y iteration)
    vectorHandler_->scal(max_d_, x_, 0, memspace_);
    vectorHandler_->scal(max_d_, y_, n_x_, memspace_);

    s_->copyFromExternal(r_yd_->getData(memspace_), memspace_, memspace_);

    if (J_d_flag_)
    {
      matrixHandler_->matvec(J_d_, x_, s_, &ONE, &MINUS_ONE, memspace_);
    }
    else
    {
      vectorHandler_->scal(MINUS_ONE, s_, memspace_);
    }

    y_d_->copyFromExternal(s_, memspace_, memspace_);
    vectorHandler_->scal(D_s_vals_, y_d_, memspace_);
    vectorHandler_->axpy(MINUS_ONE, r_s_, y_d_, memspace_);

    if (memspace_ == memory::DEVICE)
    {
      x_->syncData(memory::HOST);
      s_->syncData(memory::HOST);
      y_->syncData(memory::HOST);
      y_d_->syncData(memory::HOST);
    }
  }

  /**
   * @brief calculates the error of Ax - b
   *
   * @pre solution properly recovered using recoverSolution()
   *
   * @param[out] norm_res - Error of Ax - b
   */
  real_type hykkt::HyKKTSolver::checkError()
  {
    //  Start of block, calculate error of Ax-b
    //  Calculate error in r_x
    real_type norm_r_x_sq  = 0;
    real_type norm_rs_sq   = 0;
    real_type norm_r_y_sq  = 0;
    real_type norm_r_yd_sq = 0;
    real_type norm_resx_sq = 0;
    real_type norm_resy_sq = 0;

    // This will aggregate the squared norms of the residual and rhs
    // Note that by construction the residuals of r_s and r_yd are 0
    norm_r_x_sq  = vectorHandler_->dot(r_x_, r_x_, memspace_);
    norm_rs_sq   = vectorHandler_->dot(r_s_, r_s_, memspace_);
    norm_r_y_sq  = vectorHandler_->dot(r_y_copy_, r_y_copy_, memspace_);
    norm_r_yd_sq = vectorHandler_->dot(r_yd_, r_yd_, memspace_);

    norm_r_x_sq += norm_rs_sq + norm_r_y_sq + norm_r_yd_sq;

    matrixHandler_->matvec(H_, x_, r_x_, &MINUS_ONE, &ONE, memspace_);
    if (J_d_flag_)
    {
      matrixHandler_->matvec(J_d_tr_, y_d_, r_x_, &MINUS_ONE, &ONE, memspace_);
    }
    matrixHandler_->matvec(J_tr_copy_, y_, r_x_, &MINUS_ONE, &ONE, memspace_);
    norm_resx_sq = vectorHandler_->dot(r_x_, r_x_, memspace_);

    // Calculate error in r_y
    matrixHandler_->matvec(J_copy_, x_, r_y_copy_, &MINUS_ONE, &ONE, memspace_);
    norm_resy_sq = vectorHandler_->dot(r_y_copy_, r_y_copy_, memspace_);

    norm_resx_sq += norm_resy_sq;
    real_type norm_res = sqrt(norm_resx_sq);
    if (norm_r_x_sq > 0)
    {
      norm_res /= sqrt(norm_r_x_sq);
      printf("||Ax-b||/||b|| = %32.32g\n\n", norm_res);
    }
    else
    {
      printf("||Ax-b|| = %32.32g\n\n", norm_res);
    }

    allocated_ = true;

    return norm_res;
  }
} // namespace ReSolve
