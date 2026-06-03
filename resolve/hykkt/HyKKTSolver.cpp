/**
 * @file HyKKTSolver.cpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief HyKKT system solver implementation
 */

#include "HyKKTSolver.hpp"

#include <resolve/matrix/io.hpp>

namespace ReSolve {
  using namespace constants;

  /**
   * @brief basic constructor
   *
   * @param[in] nx   - Number of rows of the 1st row of blocks of the system
   * @param[in] md   - Number of rows of the 2nd row of blocks of the system
   * @param[in] mc   - Number of rows of the 3rd row of blocks of the system
   */
  hykkt::HyKKTSolver::HyKKTSolver(index_type nx, index_type md, index_type mc, memory::MemorySpace memspace)
    : nx_{nx},
      md_{md},
      mc_{mc},
      n_total_{nx + mc},
      m_{mc + md},
      n_{nx + md},
      N_{m_ + n_},
      memspace_{memspace}
  {
  }

  hykkt::HyKKTSolver::~HyKKTSolver()
  {
    delete rx_perm_;
    delete Hrx_perm_;
    delete schur_;
    delete Ds_vals_;
    delete ryd_scaled_;
    delete rx_til_;
    delete rx_hat_;
    delete z_;
    delete ry_copy_;
    delete J_tr_;
    delete Jd_tr_;
    delete HGam_perm_;
    delete J_perm_;
    delete J_tr_perm_;
    delete Jd_scaled_;
    delete J_copy_;
    delete J_tr_copy_;
    delete ruiz_;
    delete spgemm_htil_;
    delete spgemm_hgamma_;
    delete permutation_;
    delete cholesky_;
    delete sccg_;
  }

  /*
   * @brief loads KKT system into solver. Note: matrices and vectors,
   * including the LHS (output) vectors, must be allocated by the 
   * caller beforehand and destroyed after.
   *
   * @param file names for different components of KKT system
   *        with same nonzero structure
   *
   * @post H_, Ds_, J_, Jd_, rx_, rs_, ry_,
   *       ryd_ have new values for the system in a following
   *       solver iteration with same nonzero structure as
   *       previous iterations
   */
  void hykkt::HyKKTSolver::readMatrixFiles(
      std::istream& H_file,
      std::istream& Ds_file,
      std::istream& J_file,
      std::istream& Jd_file,
      std::istream& rx_file,
      std::istream& rs_file,
      std::istream& ry_file,
      std::istream& ryd_file)
  {
    io::updateMatrixFromFile(H_file, H_); // is_expand_symmetric?
    io::updateMatrixFromFile(Ds_file, Ds_);
    io::updateMatrixFromFile(J_file, J_);
    io::updateMatrixFromFile(Jd_file, Jd_); // Jd_tr_ will be populated later

    io::updateVectorFromFile(rx_file, rx_);
    io::updateVectorFromFile(rs_file, rs_);
    io::updateVectorFromFile(ry_file, ry_);
    io::updateVectorFromFile(ryd_file, ryd_);

    if (memspace_ == memory::DEVICE)
    {
      H_->syncData(memory::DEVICE);
      Ds_->syncData(memory::DEVICE);
      J_->syncData(memory::DEVICE);
      Jd_->syncData(memory::DEVICE);
      rx_->syncData(memory::DEVICE);
      rs_->syncData(memory::DEVICE);
      ry_->syncData(memory::DEVICE);
      ryd_->syncData(memory::DEVICE);
    }

    bool Jd_flag = Jd_->getNnz() > 0;
    status_ = (Jd_flag_ == Jd_flag); //if new nonzero structure then broken
    Jd_flag_ = Jd_flag;
  }

  /**
   * @brief Sets blocks of the KKT matrix in CSR format to user provided
   * values. It will only set pointers to user provided data; it is user's 
   * responsibility to supply and later delete that memory.
   * 
   * @param[in] H_plus_Dx - Pointer to the Hessian matrix block (nx x nx),
   * corresponding to H + Dx in the HyKKT paper.
   * @param[in] Ds - Pointter to the slack variables derivatives matrix block
   * (md x md).
   * @param[in] J - Pointer to the equality constraints Jacobian block
   * (mc x nx).
   * @param[in] Jd - Pointer to the inequality constraints Jacobian block
   * (md x nx)
   */
  void hykkt::HyKKTSolver::setMatrixBlocks(matrix::Csr* H_plus_Dx, matrix::Csr* Ds, matrix::Csr* J, matrix::Csr* Jd)
  {
    H_ = H_plus_Dx;
    Ds_ = Ds;
    J_ = J;
    Jd_ = Jd;

    bool Jd_flag = Jd->getNnz() > 0;
    // status_ = (Jd_flag_ == Jd_flag);
    status_ = true; // when using API, we can't check if sparsity pattern changed
    Jd_flag_ = Jd_flag;
  }

  /**
   * @brief Sets the blocks of the RHS vector of the system to user provided
   * values. It will only set pointers to user provided data; it is user's 
   * responsibility to supply and later delete that memory.
   *
   * @param[in] rx - Pointer to the rx vector (shape: nx)
   * @param[in] rs - Pointer to the rs vector (shape: md)
   * @param[in] ry - Pointer to the ry vector (shape: mc)
   * @param[in] ryd - Pointer to the ryd vector (shape: md)
   */
  void hykkt::HyKKTSolver::setRHSBlocks(vector::Vector* rx, vector::Vector* rs, vector::Vector* ry, vector::Vector* ryd)
  {
    rx_ = rx;
    rs_ = rs;
    ry_ = ry;
    ryd_ = ryd;
  }

  /**
   * @brief Sets the pointers to the blocks of the LHS (output) vector of the 
   * system. It will set pointers to the vector object that will contain the 
   * solver's solutions. Existing values will not be used and will be overridden.
   * It is user's responsibility to supply and later delete that memory.
   *
   * @param[in] x - Pointer to the x vector (shape: nx)
   * @param[in] s - Pointer to the s vector (shape: md)
   * @param[in] y - Pointer to the y vector (shape: mc)
   * @param[in] yd - Pointer to the yd vector (shape: md)
   */
  void hykkt::HyKKTSolver::setLHSPointers(vector::Vector* x, vector::Vector* s, vector::Vector* y, vector::Vector* yd)
  {
    x_ = x;
    s_ = s;
    y_ = y;
    yd_ = yd;
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
    // TODO: Review sparsity pattern checking in HyKKT
    if(!status_ && allocated_){
      printf("\n\nERROR: USING HYKKT WITH NEW NONZERO STRUCTURE\n\n");
      std::cout << "status = "      << status_ 
                << ", allocated = " << allocated_
                << "\n";
      return 1;
    }

    setupParameters();
    
    if(!allocated_){
      setupSpGEMMHtil();
    }
    computeSpGEMMHtil();
    
    setupSolutionCheck();

    if(!allocated_){
      setupRuizScaling();
    }
    computeRuizScaling();

    if(!allocated_){
      setupSpGEMMHGamma();
    }
    computeSpGEMMHGamma();

    if(!allocated_){
      setupPermutation();
    }
    applyPermutation();

    if(!allocated_){
      setupHGammaFactorization();
    }
    computeHGammaFactorization();
  
    setupConjugateGradient();
    computeConjugateGradient();

    recoverSolution();
    return checkError();
  }

  /**
   * @brief allocates and initiates variables for KKT system
   *
   * @pre jd_flag_ determines if variables used for Spgemm Htil
   *      should be initiated
   *
   * @post all variables used for hykkt are allocated for; Jd-
   *       related variables are not initiated if Jd nnz == 0
   */
  void hykkt::HyKKTSolver::setupParameters()
  {
    // Assume all matrix blocks and RHS blocks are already set
    
    std::cout << "H size: " << H_->getNumRows() << " "<< H_->getNumColumns() << " "<< H_->getNnz() << " \n";
    std::cout << "J size: " << J_->getNumRows() << "  " << J_->getNumColumns() << "  "<< J_->getNnz() << " \n";
    std::cout << "Ds nnz = " << Ds_->getNnz() << "\n";

    if (!allocated_)
    {
      rx_perm_   = new vector::Vector(nx_);
      Hrx_perm_  = new vector::Vector(nx_);
      schur_ = new vector::Vector(mc_);
      Ds_vals_ = new vector::Vector(md_);
      ryd_scaled_ = new vector::Vector(ryd_->getSize());
      rx_til_ = new vector::Vector(nx_);
      rx_hat_ = new vector::Vector(nx_);
      z_ = new vector::Vector(nx_);
      ry_copy_ = new vector::Vector(mc_);
      J_tr_ = new matrix::Csr(J_->getNumColumns(), J_->getNumRows(), J_->getNnz());
      Jd_tr_ = new matrix::Csr(Jd_->getNumColumns(), Jd_->getNumRows(), Jd_->getNnz());
      Htil_ = new matrix::Csr(H_->getNumRows(), H_->getNumColumns(), H_->getNnz());
      J_perm_ = new matrix::Csr(J_->getNumRows(), J_->getNumColumns(), J_->getNnz());
      J_tr_perm_ = new matrix::Csr(J_tr_->getNumRows(), J_tr_->getNumColumns(), J_tr_->getNnz());
      Jd_scaled_ = new matrix::Csr(Jd_->getNumRows(), Jd_->getNumColumns(), Jd_->getNnz());

      Ds_vals_->setData(Ds_->getValues(memspace_), memspace_);
      rx_perm_->allocate(memspace_);
      Hrx_perm_->allocate(memspace_);
      schur_->allocate(memspace_);
      rx_til_->allocate(memspace_);
      rx_hat_->allocate(memspace_);
      z_->allocate(memspace_);
      J_tr_->allocateMatrixData(memspace_);
      Jd_tr_->allocateMatrixData(memspace_);
      J_tr_perm_->allocateMatrixData(memory::HOST);
      J_perm_->allocateMatrixData(memory::HOST);
      Jd_scaled_->allocateWithExternalSparsityPattern(Jd_->getRowData(memspace_), Jd_->getColData(memspace_), Jd_->getNnz(), memspace_);
    }
    else if (memspace_ == memory::DEVICE)
    {
      Jd_->syncData(memory::DEVICE); // check if this is redundant
    }

    ry_copy_->copyFromExternal(ry_, memspace_, memspace_);
    
    // check if this is redundant in later iterations
    matrixHandler_->transpose(J_, J_tr_, memspace_);
    if (Jd_flag_)
    {
    matrixHandler_->transpose(Jd_, Jd_tr_, memspace_);
      Jd_scaled_->copyValues(Jd_->getValues(memspace_), memspace_, memspace_);
    }
  }

  /**
   * @brief Creates the SpGEMM solver for the Htilda matrix and
   * loads the result matrix pointer
   * @post The result of the solver is set to be written into Htil_
   */
  void hykkt::HyKKTSolver::setupSpGEMMHtil()
  {
    spgemm_htil_ = new SpGEMM(memspace_, ONE, ONE);
    spgemm_htil_->loadResultMatrix(&Htil_); // Htil_ will be created by SpGEMM at this step
  }

  /*
   * @brief computes SpGEMM to calculate Htilda matrix
  *
   * @pre matrices and spgemm_htil_ properly allocated using setup
   *      method for spgemm_htil_
  *
   * @post Htilda calculated using Jd matrix if Jd nnz > 0 and
   *       is set to H if Jd nnz == 0
  */
  void hykkt::HyKKTSolver::computeSpGEMMHtil()
  {
    if (Jd_flag_)
    {
      matrixHandler_->leftScale(Ds_vals_, Jd_scaled_, memspace_);
      spgemm_htil_->loadProductMatrices(Jd_tr_, Jd_scaled_);
      spgemm_htil_->loadSumMatrix(H_);

      ryd_scaled_->copyFromExternal(ryd_, memspace_, memspace_);
      vectorHandler_->scal(Ds_vals_, ryd_scaled_, memspace_);

      // ryd_scaled_ = rs + Ds * ryd
      vectorHandler_->axpy(ONE, rs_, ryd_scaled_, memspace_);
      rx_til_->copyFromExternal(rx_, memspace_, memspace_);
      matrixHandler_->matvec(Jd_tr_, ryd_scaled_, rx_til_, &ONE, &ONE, memspace_);
      spgemm_htil_->compute();
    }
    else
    {
      Htil_->copyFromExternal(H_->getRowData(memspace_),
                              H_->getColData(memspace_),
                              H_->getValues(memspace_),
                              H_->getNnz(),
                              memspace_,
                              memspace_);

      rx_til_->copyFromExternal(rx_, memspace_, memspace_);
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
      J_tr_copy_ = new matrix::Csr(J_tr_->getNumRows(), J_tr_->getNumColumns(), J_tr_->getNnz());
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
    ruiz_ = new RuizScaling(nx_, n_total_, memspace_);
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
    ruiz_->addMatrixData(Htil_, J_, J_tr_);
    ruiz_->addRhsData(rx_til_, ry_);
    ruiz_->scale(ruiz_its_);
    max_d_ = ruiz_->getAggregateScalingVector();
  }
  
  /**
   * @brief Creates the SpGEMM solver for the HGamma matrix and
   * loads the result matrix pointer
   * @post The result of the solver is set to be written into HGam_
   */
  void hykkt::HyKKTSolver::setupSpGEMMHGamma()
  {
    spgemm_hgamma_ = new SpGEMM(memspace_, gamma_, ONE);
    spgemm_hgamma_->loadProductMatrices(J_tr_, J_);
    spgemm_hgamma_->loadSumMatrix(Htil_);
    spgemm_hgamma_->loadResultMatrix(&HGam_); // HGam_ will be created by SpGEMM at this step

  }

  /*
  * @brief computes SpGEMM to calculate HGamma matrix
  *
  * @pre matrices and spgemm_hgamma properly allocated using setup
  *      method for spgemm_hgamma
  *
  * @post HGamma CSR now represents J_tr * J + Htil
  */
  void hykkt::HyKKTSolver::computeSpGEMMHGamma()
  {
    spgemm_hgamma_->compute();
    rx_hat_->copyFromExternal(rx_til_, memspace_, memspace_);
    matrixHandler_->matvec(J_tr_, ry_, rx_hat_, &gamma_, &ONE, memspace_);
  }
  
  void hykkt::HyKKTSolver::setupPermutation()
  {
    HGam_perm_ = new matrix::Csr(nx_, nx_, HGam_->getNnz()); // Nnz not known at setupParameters(), so we have to create it here
    HGam_perm_->allocateMatrixData(memory::HOST);

    if (memspace_ == memory::DEVICE)
    {
      HGam_->syncData(memory::HOST);
    }

    // These permutation steps are device-only
    permutation_ = new Permutation(nx_, mc_, HGam_->getNnz(), J_->getNnz(), memspace_);
    permutation_->addMatrixInfo(HGam_, J_, J_tr_);
    permutation_->symAmd();
    permutation_->invertPerm();

    permutation_->vecMapRC(HGam_perm_->getRowData(memory::HOST), HGam_perm_->getColData(memory::HOST));
    HGam_perm_->setUpdated(memory::HOST);
    index_type* j_rows      = J_->getRowData(memory::HOST);
    index_type* j_perm_rows = J_perm_->getRowData(memory::HOST);

    for (index_type i = 0; i <= J_->getNumRows(); ++i)
    {
      j_perm_rows[i] = j_rows[i];
    }

    permutation_->vecMapC(J_perm_->getColData(memory::HOST));
    J_perm_->setUpdated(memory::HOST);

    if (memspace_ == memory::DEVICE)
    {
      J_tr_->syncData(memory::HOST);
    }

    permutation_->vecMapR(J_tr_perm_->getRowData(memory::HOST), J_tr_perm_->getColData(memory::HOST));

    if (memspace_ == memory::DEVICE)
    {
      HGam_perm_->syncData(memory::DEVICE);
      J_perm_->syncData(memory::DEVICE);
      J_tr_perm_->syncData(memory::DEVICE);
    }
  }

  /* 
  * @brief Applies permutations to values of HGam, J, J^T matrices
            and rx_hat vector
  *
  * @pre Permutation maps for the matrices and vector
  *      computed using setupPermutation()
  *      
  *
  * @post HGam_perm_, J_perm_, J_tr_perm_, rx_perm_ now contain permuted
  *       values of HGam_, J_, J_tr_, rx_hat_
  */
  void hykkt::HyKKTSolver::applyPermutation()
  {
    permutation_->mapIndex(PERM_HES_V,
                          HGam_->getValues(memspace_),
                          HGam_perm_->getValues(memspace_));
    permutation_->mapIndex(PERM_JAC_V,
                           J_->getValues(memspace_),
                           J_perm_->getValues(memspace_));
    permutation_->mapIndex(PERM_JAC_TR_V,
                           J_tr_->getValues(memspace_),
                           J_tr_perm_->getValues(memspace_));
    permutation_->mapIndex(PERM_V,
                             rx_hat_->getData(memspace_),
                             rx_perm_->getData(memspace_));
    rx_perm_->setDataUpdated(memspace_);
  }
  
  void hykkt::HyKKTSolver::setupHGammaFactorization()
  {
    cholesky_ = new CholeskySolver(memspace_);
    cholesky_->addMatrixInfo(HGam_perm_);
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
   * @post HGamma numerical factorization is computed, thus updating
   *       HGam_perm_
   */
  void hykkt::HyKKTSolver::computeHGammaFactorization()
  {
    cholesky_->numericalFactorization();
  }
  
  void hykkt::HyKKTSolver::setupConjugateGradient()
  {
    cholesky_->solve(Hrx_perm_, rx_perm_);
    schur_->copyFromExternal(ry_, memspace_, memspace_);
    matrixHandler_->matvec(J_perm_, Hrx_perm_, schur_, &ONE, &MINUS_ONE, memspace_);
    
      sccg_ = new SchurComplementConjugateGradient(J_->getNumRows(), J_->getNumColumns(), cholesky_, memspace_, *matrixHandler_, *vectorHandler_);
    }
    sccg_->addMatrixInfo(J_perm_, J_tr_perm_);
    y_->setToZero(memspace_);
    sccg_->addVectorInfo(y_, schur_);
    sccg_->setup();
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
   * @post rx_, rs_, ry_copy_, and ryd_ contain the solution
   *       on the device
   */
  void hykkt::HyKKTSolver::recoverSolution()
  {
    matrixHandler_->matvec(J_tr_perm_,
                           y_,
                           rx_perm_,
                           &MINUS_ONE,
                           &ONE,
                           memspace_);
    // block-recovering the solution to the original system by parts
    // this part is to recover delta_x
    cholesky_->solve(z_, rx_perm_);
    permutation_->mapIndex(REV_PERM_V, z_->getData(memspace_), x_->getData(memspace_));
    x_->setDataUpdated(memspace_);
    
    // scale back delta_y and delta_x (every iteration)
    vectorHandler_->scal(max_d_, x_, 0, nx_, memspace_);
    vectorHandler_->scal(max_d_, y_, nx_, n_total_, memspace_);

    s_->copyFromExternal(ryd_->getData(memspace_), memspace_, memspace_);

    if (Jd_flag_)
    {
      matrixHandler_->matvec(Jd_, x_, s_, &ONE, &MINUS_ONE, memspace_);
    }
    else
    {
      vectorHandler_->scal(MINUS_ONE, s_, memspace_);
    }

    yd_->copyFromExternal(s_, memspace_, memspace_);
    vectorHandler_->scal(Ds_vals_, yd_, memspace_);
    vectorHandler_->axpy(MINUS_ONE, rs_, yd_, memspace_);

    if (memspace_ == memory::DEVICE)
    {
      x_->syncData(memory::HOST);
      s_->syncData(memory::HOST);
      y_->syncData(memory::HOST);
      yd_->syncData(memory::HOST);
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
    //  Calculate error in rx
    real_type norm_rx_sq   = 0;
    real_type norm_rs_sq   = 0;
    real_type norm_ry_sq   = 0;
    real_type norm_ryd_sq  = 0;
    real_type norm_resx_sq = 0;
    real_type norm_resy_sq = 0; 
    
    // This will aggregate the squared norms of the residual and rhs
    // Note that by construction the residuals of rs and ryd are 0
    norm_rx_sq = vectorHandler_->dot(rx_, rx_, memspace_);
    norm_rs_sq = vectorHandler_->dot(rs_, rs_, memspace_);
    norm_ry_sq = vectorHandler_->dot(ry_copy_, ry_copy_, memspace_);
    norm_ryd_sq = vectorHandler_->dot(ryd_, ryd_, memspace_);

    norm_rx_sq += norm_rs_sq + norm_ry_sq + norm_ryd_sq;
    
    matrixHandler_->matvec(H_, x_, rx_, &MINUS_ONE, &ONE, memspace_);
    if (Jd_flag_)
    {
      matrixHandler_->matvec(Jd_tr_, yd_, rx_, &MINUS_ONE, &ONE, memspace_);
    }
    matrixHandler_->matvec(J_tr_copy_, y_, rx_, &MINUS_ONE, &ONE, memspace_);
    norm_resx_sq = vectorHandler_->dot(rx_, rx_, memspace_);

    // Calculate error in ry
    matrixHandler_->matvec(J_copy_, x_, ry_copy_, &MINUS_ONE, &ONE, memspace_);
    norm_resy_sq = vectorHandler_->dot(ry_copy_, ry_copy_, memspace_);

    // Calculate final relative norm
    norm_resx_sq += norm_resy_sq;
    real_type norm_res = sqrt(norm_resx_sq) / sqrt(norm_rx_sq);
    printf("||Ax-b||/||b|| = %32.32g\n\n", norm_res);

    allocated_ = true;
    
    return norm_res;
  }
}