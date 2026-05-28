#include "HyKKTSolver.hpp"

#include <resolve/matrix/io.hpp>

namespace ReSolve {
  /**
   * @brief basic constructor
   *
   * @param[in] nx   - 
   * @param[in] mc   - 
   * @param[in] nd   -
   */
  hykkt::HyKKTSolver::HyKKTSolver(index_type nx, index_type mc, index_type md, memory::MemorySpace memspace)
    : mc_{mc},
      md_{md},
      nx_{nx},
      n_total_{nx + mc},
      m_{mc + md},
      n_{nx + md},
      N_{m_ + n_},
      memspace_{memspace}
  {
  }

  using namespace constants;

  /*
   * @brief loads KKT system into solver. Note: matrices and vectors,
   * including the LHS (output) vectors, must be allocated by the 
   * caller beforehand and destroyed after.
   *
   * @param file names for different components of KKT system
   *        with same nonzero structure
   *
   * @post H_, Dx_, Ds_, J_, Jd_, rx_, rs_, ry_,
   *       ryd_ have new values for the system in a following
   *       solver iteration with same nonzero structure as
   *       previous iterations
   */
  void hykkt::HyKKTSolver::readMatrixFiles( // SHOULD THIS CLASS OWN THE MATRICES/VECTORS?
      std::istream& H_file,
      std::istream& Dx_file,
      std::istream& Ds_file,
      std::istream& J_file,
      std::istream& Jd_file,
      std::istream& rx_file,
      std::istream& rs_file,
      std::istream& ry_file,
      std::istream& ryd_file)
  {
    io::updateMatrixFromFile(H_file, H_); // is_expand_symmetric?
    io::updateMatrixFromFile(Dx_file, Dx_);
    io::updateMatrixFromFile(Ds_file, Ds_);
    io::updateMatrixFromFile(J_file, J_);
    io::createCsrFromFile(Jd_file, Jd_); // Jd_tr_ will be populated later

    io::updateVectorFromFile(rx_file, rx_);
    io::updateVectorFromFile(rs_file, rs_);
    io::updateVectorFromFile(ry_file, ry_);
    io::updateVectorFromFile(ryd_file, ryd_);

    if (memspace_ == memory::DEVICE)
    {
      H_->syncData(memory::DEVICE);
      Dx_->syncData(memory::DEVICE);
      Ds_->syncData(memory::DEVICE);
      J_->syncData(memory::DEVICE);
      J_tr_->syncData(memory::DEVICE);
      Jd_->syncData(memory::DEVICE);
      Jd_tr_->syncData(memory::DEVICE);
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
   * @brief Sets Hessian matrix block (square, nx x nx) in to user 
   * provided values. It will only set pointers to user provided 
   * data; it is user's responsibility to supply and later delete 
   * that data.
   * 
   * @param[in] H - Pointer to the H matrix in CSR format.
   */
  void hykkt::HyKKTSolver::set_H(matrix::Csr* H)
  {
    H_ = H;
  }

  /**
   * @brief Sets slack variables derivatives matrix block (square, 
   * nx x nx) in to user provided values. It will only set pointers 
   * to user provided data; it is user's responsibility to supply 
   * and later delete that data.
   * 
   * @param[in] Ds - Pointer to the Ds matrix in CSR format.
   */
  void hykkt::HyKKTSolver::set_Ds(matrix::Csr* Ds)
  {
    Ds_ = Ds;
  }

  /**
   * @brief Sets equality constraints Jacobian block (mc x nx)
   * in to user provided values. It will only set pointers to 
   * user provided data; it is user's responsibility to supply 
   * and later delete that data.
   * 
   * @param[in] J - Pointer to the J matrix in CSR format.
   * @param[in] J_tr - Pointer to the transposed J matrix in CSR format.
   */
  void hykkt::HyKKTSolver::set_J(matrix::Csr* J, matrix::Csr* J_tr)
  {
    J_ = J;
    J_tr_ = J_tr;
  }

  /**
   * @brief Sets inequality constraints Jacobian block (md x nx) in to user
   * provided values. It will only set pointers to user provided data; it 
   * is user's responsibility to supply and later delete that data.
   * 
   * @param[in] Jd - Pointer to the Jd matrix in CSR format.
   * @param[in] Jd_tr - Pointer to the transposed Jd matrix in CSR format.
   */
  void hykkt::HyKKTSolver::set_Jd(matrix::Csr* Jd, matrix::Csr* Jd_tr)
  {
    Jd_ = Jd;
    Jd_tr_ = Jd_tr;
    // bool jd_flag = mat_jd_.nnz_ > 0;
    // // status_ = (jd_flag_ == jd_flag);
    // status_ = true; // when using API, we can't check if sparsity pattern changed
    // jd_flag_ = jd_flag;
  }

  void hykkt::HyKKTSolver::set_rx(vector::Vector* rx)
  {
    rx_ = rx;
  }

  void hykkt::HyKKTSolver::set_rs(vector::Vector* rs)
  {
    rs_ = rs;
  }

  void hykkt::HyKKTSolver::set_ry(vector::Vector* ry)
  {
    ry_ = ry;
  }

  void hykkt::HyKKTSolver::set_ryd(vector::Vector* ryd)
  {
    ryd_ = ryd;
  }

  void hykkt::HyKKTSolver::set_x(vector::Vector* x)
  {
    x_ = x;
  }

  void hykkt::HyKKTSolver::set_s(vector::Vector* s)
  {
    s_ = s;
  }

  void hykkt::HyKKTSolver::set_y(vector::Vector* y)
  {
    y_ = y;
  }

  void hykkt::HyKKTSolver::set_yd(vector::Vector* yd)
  {
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
   * @param[out] - Boolean that indicates whether the solve was successful
   *
   * @post solution to given KKT system is computed using Hykkt
  */
  int hykkt::HyKKTSolver::solve()
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
   * @post all variables used for hykkt are allocated for; jd
   *       related variables are not initiated if JD nnz == 0
   */
  void hykkt::HyKKTSolver::setupParameters()
  {
    // Assume all matrix blocks and RHS blocks are already set
    
    std::cout << "H size: " << H_->getNumRows() << " "<< H_->getNumColumns() << " "<< H_->getNnz() << " \n";
    std::cout << "J size: " << J_->getNumRows() << "  " << J_->getNumRows() << "  "<< J_->getNumRows() << " \n";
    std::cout << "Ds nnz = " << Ds_->getNnz() << "\n";

    if (!allocated_)
    {
      rxp_   = new vector::Vector(nx_);
      hrxp_  = new vector::Vector(nx_);
      schur_ = new vector::Vector(mc_);
      Ds_vals_ = new vector::Vector(md_);
      ryd_scaled_ = new vector::Vector(ryd_->getSize());
      rx_til_ = new vector::Vector(nx_);
      rx_hat_ = new vector::Vector(nx_);
      z_ = new vector::Vector(nx_);
      ry_copy_ = new vector::Vector(mc_);
      Htil_ = new matrix::Csr(H_->getNumRows(), H_->getNumColumns(), H_->getNnz());
      J_perm_ = new matrix::Csr(J_->getNumRows(), J_->getNumColumns(), J_->getNnz());
      J_tr_perm_ = new matrix::Csr(J_tr_->getNumRows(), J_tr_->getNumColumns(), J_tr_->getNnz());
      Jd_scaled_ = new matrix::Csr(Jd_->getNumRows(), Jd_->getNumColumns(), Jd_->getNnz());

      Ds_vals_->setData(Ds_->getValues(memspace_), memspace_);
      rxp_->allocate(memspace_);
      hrxp_->allocate(memspace_);
      schur_->allocate(memspace_);
      Ds_vals_->allocate(memspace_);
      rx_til_->allocate(memspace_);
      rx_hat_->allocate(memspace_);
      z_->allocate(memspace_);
      J_perm_->allocateMatrixData(memspace_);
      J_tr_perm_->allocateMatrixData(memspace_);
    }
    else if (memspace_ == memory::DEVICE)
    {
      // nonzero structure stays same so only update values each iteration
      // IS THIS NEEDED? COME BACK LATER. DO H_ AND J_ DATA STAY ON DEVICE?
    }

    if (Jd_flag_)
    {
      // COME BACK LATER
    }

    // SOME COPIES
    
    matrixHandler_->transpose(J_, J_tr_, memspace_);
  }

  void hykkt::HyKKTSolver::setupSpGEMMHtil()
  {
    spgemm_htil_ = new SpGEMM(memspace_, ONE, ONE);
    spgemm_htil_->loadProductMatrices(Jd_tr_, Jd_scaled_);
    spgemm_htil_->loadSumMatrix(H_);
    spgemm_htil_->loadResultMatrix(&Htil_); // Htil_ will be created by SpGEMM at this step
  }

  void hykkt::HyKKTSolver::setupRuizScaling()
  {
    ruiz_ = new RuizScaling(nx_, n_total_, memspace_);
  }

  void hykkt::HyKKTSolver::setupSpGEMMHGamma()
  {
    spgemm_gamma_ = new SpGEMM(memspace_, gamma_, ONE);
    spgemm_gamma_->loadProductMatrices(J_tr_, J_);
    spgemm_gamma_->loadSumMatrix(Htil_);
    spgemm_gamma_->loadResultMatrix(&HGam_); // HGam_ will be created by SpGEMM at this step
  }

  void hykkt::HyKKTSolver::setupPermutation()
  {
    HGam_perm_ = new matrix::Csr(nx_ + 1, nx_ + 1, HGam_->getNnz()); // Nnz not known at setupParameters(), so we have to create it here
    HGam_perm_->allocateMatrixData(memory::DEVICE);
    
    if (memspace_ == memory::DEVICE)
    {
      HGam_->syncData(memory::HOST);
    }

    // These permutation steps are device-only
    permutation_ = new Permutation(nx_, mc_, H_->getNnz(), J_->getNnz(), memspace_);
    permutation_->addMatrixInfo(H_, J_, J_tr_);
    permutation_->symAmd();
    permutation_->invertPerm();

    permutation_->vecMapRC(HGam_perm_->getRowData(memspace_), HGam_perm_->getColData(memspace_));
    permutation_->vecMapC(J_perm_->getColData(memspace_));

    if (memspace_ == memory::DEVICE)
    {
      J_tr_->syncData(memory::HOST);
    }

    permutation_->vecMapR(J_tr_perm_->getRowData(memspace_), J_tr_perm_->getColData(memspace_));

    if (memspace_ == memory::DEVICE)
    {
      HGam_perm_->syncData(memory::DEVICE);
      J_perm_->syncData(memory::DEVICE);
      J_tr_perm_->syncData(memory::DEVICE);
    }
  }

  void hykkt::HyKKTSolver::setupHGammaFactorization()
  {
    cholesky_ = new CholeskySolver(memspace_);
    cholesky_->addMatrixInfo(HGam_perm_);
    cholesky_->symbolicAnalysis();
    cholesky_->setPivotTolerance(tol_);
  }

  void hykkt::HyKKTSolver::setupConjugateGradient()
  {
    cholesky_->solve(hrxp_, rxp_);
    schur_->setData(ry_->getData(memspace_), memspace_);
    matrixHandler_->matvec(J_perm_, hrxp_, schur_, &ONE, &MINUS_ONE, memspace_);
    
    if (!allocated_)
    {
      sccg_ = new SchurComplementConjugateGradient(J_->getNumRows(), J_->getNumColumns(), cholesky_, memspace_, *matrixHandler_, *vectorHandler_);
    }
    sccg_->addMatrixInfo(J_, J_tr_);
    sccg_->addVectorInfo(y_, schur_);
    sccg_->setup();
  }

  /*
  * @brief computes Spgemm to calculate Htilda matrix
  *
  * @pre matrices and sc_til_ properly allocated using setup
  *      method for spgemm_htil
  *
  * @post Htilda calculated using JD matrix if JD nnz > 0 and
  *       is set to H if JD nnz == 0
  */
  void hykkt::HyKKTSolver::computeSpGEMMHtil()
  {
    if (Jd_flag_)
    {
      matrixHandler_->transpose(Jd_, Jd_tr_, memspace_);
      
      Jd_scaled_->setDataPointers(Jd_->getRowData(memspace_), Jd_->getColData(memspace_), nullptr, memspace_);
      Jd_scaled_->copyValues(Jd_->getValues(memspace_), memspace_, memspace_);
      matrixHandler_->leftScale(Ds_vals_, Jd_scaled_, memspace_);

      ryd_scaled_->copyFromExternal(ryd_, memspace_, memspace_);
      vectorHandler_->scal(Ds_vals_, ryd_scaled_, memspace_);
      vectorHandler_->axpy(ONE, ryd_scaled_, rs_, memspace_);
      matrixHandler_->matvec(Jd_tr_, ryd_scaled_, rx_til_, &ONE, &ONE, memspace_); // huh???? rx_til_ never initialized? not in old version either
      spgemm_htil_->compute(); // it seems like this is the replacement for reuse(), but double check that it's the same
    }
    else
    {
      Htil_->copyFromExternal(H_->getRowData(memspace_),
                              H_->getColData(memspace_),
                              H_->getValues(memspace_),
                              H_->getNnz(),
                              memspace_,
                              memspace_);
    }
  }
  
  /*
  * @brief computes ruiz scaling so we can judge the size of
  *        gamma and delta min relative to H gammma system
  *
  * @pre matrices and ri_sssssssss allocated using setup method
  *      for ruiz_scaling 
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
  
  /*
  * @brief computes Spgemm to calculate HGamma matrix
  *
  * @pre matrices and sc_gamma_ properly allocated using setup
  *      method for spgemm_hgamma
  *
  * @post HGamma CSR now represent jc_t_desc_*jc_desc_ + htil
  */
  void hykkt::HyKKTSolver::computeSpGEMMHGamma()
  {
    spgemm_gamma_->compute();
    rx_hat_->copyFromExternal(rx_til_, memspace_, memspace_);
    matrixHandler_->matvec(J_tr_, ry_, rx_hat_, &gamma_, &ONE, memspace_);
  }
  
  /* 
  * @brief Applies permutations to values of Hgam, Jc, Jc^T matrices 
            and d_rx_hat vector
  *
  * @pre Permutation maps for the matrices and vector
  *      computed using setup_permutation()
  *      
  *
  * @post hgam_v_p_, jc_v_p_, jct_v_p_, d_rxp_ are now permuted
  *       values of hgam_v_, jc_v_, jc_t_v_, d_rx_hat_
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
  }
  
  /**
   * @brief sparse Cholesky factorization on permuted (1,1) block
   *        so that LDLt does not have to be used
   *
   * @pre symbolic analysis already computed using setup method
   *      for hgamma_factorization
   *
   * @post hgamma numerical factorization is computed, thus updating
   *       hgam_v_p_
   */
  void hykkt::HyKKTSolver::computeHGammaFactorization()
  {
    cholesky_->numericalFactorization();
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
   * @brief recovers solution from hykkt solver
   *
   * @pre execute functions setup and computed correctly
   *
   * @post d_rx_, d_rs_, d_ry_copy_, and d_ryd_ contain the solution
   *       on the device
   */
  void hykkt::HyKKTSolver::recoverSolution()
  {
    matrixHandler_->matvec(J_tr_,
                           y_,
                           rxp_,
                           &MINUS_ONE,
                           &ONE,
                           memspace_);
    // block-recovering the solution to the original system by parts
    // this part is to recover delta_x
    cholesky_->solve(z_, rxp_);
    permutation_->mapIndex(REV_PERM_V, z_->getData(memspace_), x_->getData(memspace_));
    
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
   * @pre solution properly recovered using recover_solution()
   * 
   * @param[out] Boolean - 0 if error small enough, 1 if hykkt failed
   *
   * @post solver status = success if error is smaller than
   *       optimization solver error
   */
  int hykkt::HyKKTSolver::checkError()
  {
    //  Start of block, calculate error of Ax-b 
    //  Calculate error in rx
    double norm_rx_sq   = 0;
    double norm_rs_sq   = 0;
    double norm_ry_sq   = 0;
    double norm_ryd_sq  = 0;
    double norm_resx_sq = 0;
    double norm_resy_sq = 0; 
    
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
    matrixHandler_->matvec(J_tr_, y_, rx_, &MINUS_ONE, &ONE, memspace_);
    norm_resx_sq = vectorHandler_->dot(rx_, rx_, memspace_);

    // Calculate error in ry
    matrixHandler_->matvec(J_copy_, x_, ry_copy_, &MINUS_ONE, &ONE, memspace_);
    norm_resy_sq = vectorHandler_->dot(ry_copy_, ry_copy_, memspace_);

    // Calculate final relative norm
    norm_resx_sq += norm_resy_sq;
    double norm_res = sqrt(norm_resx_sq) / sqrt(norm_rx_sq);
    printf("||Ax-b||/||b|| = %32.32g\n\n", norm_res);

    allocated_ = true;
    
    if (norm_res < norm_tol_)
    {
      printf("Residual test passed\n");
      return 0;
    }
    else
    {
      printf("Residual test failed\n");
      return 1;
    }
  }
}