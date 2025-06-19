#include "LinSolverDirectCuSolverGLU.hpp"

#include <cstring> // includes memcpy
#include <vector>

#include <resolve/Profiling.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{
  using vector_type = vector::Vector;
  using out         = io::Logger;

  LinSolverDirectCuSolverGLU::LinSolverDirectCuSolverGLU(LinAlgWorkspaceCUDA* workspace)
  {
    this->workspace_ = workspace;
  }

  LinSolverDirectCuSolverGLU::~LinSolverDirectCuSolverGLU()
  {
    mem_.deleteOnDevice(glu_buffer_);
    cusparseDestroyMatDescr(descr_M_);
    cusparseDestroyMatDescr(descr_A_);
    cusolverSpDestroyGluInfo(info_M_);
    delete M_;
  }

  int LinSolverDirectCuSolverGLU::setup(matrix::Sparse* A,
                                        matrix::Sparse* L,
                                        matrix::Sparse* U,
                                        index_type*     P,
                                        index_type*     Q,
                                        vector_type* /** rhs */)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    int error_sum = 0;

    LinAlgWorkspaceCUDA* workspaceCUDA = workspace_;
    // get the handle
    handle_cusolversp_ = workspaceCUDA->getCusolverSpHandle();
    A_                 = (matrix::Csr*) A;
    index_type n       = A_->getNumRows();
    index_type nnz     = A_->getNnz();
    // create combined factor
    combineFactors(L, U);

    // set up descriptors
    cusparseCreateMatDescr(&descr_M_);
    cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);
    cusolverSpCreateGluInfo(&info_M_);

    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

    // set up the GLU
    status_cusolver_ = cusolverSpDgluSetup(handle_cusolversp_,
                                           n,
                                           nnz,
                                           descr_A_,
                                           A_->getRowData(memory::HOST),
                                           A_->getColData(memory::HOST),
                                           P,            /** base-0 */
                                           Q,            /** base-0 */
                                           M_->getNnz(), /** nnzM */
                                           descr_M_,
                                           M_->getRowData(memory::HOST),
                                           M_->getColData(memory::HOST),
                                           info_M_);
    error_sum += status_cusolver_;
    // NOW the buffer
    size_t buffer_size;
    status_cusolver_ = cusolverSpDgluBufferSize(handle_cusolversp_, info_M_, &buffer_size);
    error_sum += status_cusolver_;

    mem_.allocateBufferOnDevice(&glu_buffer_, buffer_size);

    status_cusolver_ = cusolverSpDgluAnalysis(handle_cusolversp_, info_M_, glu_buffer_);
    error_sum += status_cusolver_;

    // reset and refactor so factors are ON THE GPU

    status_cusolver_ = cusolverSpDgluReset(handle_cusolversp_,
                                           n,
                                           /** A is original matrix */
                                           nnz,
                                           descr_A_,
                                           A_->getValues(memory::DEVICE),
                                           A_->getRowData(memory::DEVICE),
                                           A_->getColData(memory::DEVICE),
                                           info_M_);
    error_sum += status_cusolver_;

    status_cusolver_ = cusolverSpDgluFactor(handle_cusolversp_, info_M_, glu_buffer_);
    error_sum += status_cusolver_;

    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  void LinSolverDirectCuSolverGLU::combineFactors(matrix::Sparse* L, matrix::Sparse* U)
  {
    // L and U need to be in CSC format
    index_type  n    = L->getNumRows();
    index_type* Lp   = L->getColData(memory::HOST);
    index_type* Li   = L->getRowData(memory::HOST);
    index_type* Up   = U->getColData(memory::HOST);
    index_type* Ui   = U->getRowData(memory::HOST);
    index_type  nnzM = (L->getNnz() + U->getNnz() - n);
    M_               = new matrix::Csr(n, n, nnzM);
    M_->allocateMatrixData(memory::HOST);
    index_type* mia = M_->getRowData(memory::HOST);
    index_type* mja = M_->getColData(memory::HOST);
    index_type  row;
    for (index_type i = 0; i < n; ++i)
    {
      // go through EACH COLUMN OF L first
      for (index_type j = Lp[i]; j < Lp[i + 1]; ++j)
      {
        row = Li[j];
        // BUT dont count diagonal twice, important
        if (row != i)
        {
          mia[row + 1]++;
        }
      }
      // then each column of U
      for (index_type j = Up[i]; j < Up[i + 1]; ++j)
      {
        row = Ui[j];
        mia[row + 1]++;
      }
    }
    // then organize mia_;
    mia[0] = 0;
    for (index_type i = 1; i < n + 1; i++)
    {
      mia[i] += mia[i - 1];
    }

    std::vector<int> Mshifts(n, 0);
    for (index_type i = 0; i < n; ++i)
    {
      // go through EACH COLUMN OF L first
      for (int j = Lp[i]; j < Lp[i + 1]; ++j)
      {
        row = Li[j];
        if (row != i)
        {
          // place (row, i) where it belongs!
          mja[mia[row] + Mshifts[row]] = i;
          Mshifts[row]++;
        }
      }
      // each column of U next
      for (index_type j = Up[i]; j < Up[i + 1]; ++j)
      {
        row                          = Ui[j];
        mja[mia[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
  }

  int LinSolverDirectCuSolverGLU::refactorize()
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    int error_sum    = 0;
    status_cusolver_ = cusolverSpDgluReset(handle_cusolversp_,
                                           A_->getNumRows(),
                                           /** A is original matrix */
                                           A_->getNnz(),
                                           descr_A_,
                                           A_->getValues(memory::DEVICE),
                                           A_->getRowData(memory::DEVICE),
                                           A_->getColData(memory::DEVICE),
                                           info_M_);
    error_sum += status_cusolver_;

    status_cusolver_ = cusolverSpDgluFactor(handle_cusolversp_, info_M_, glu_buffer_);
    error_sum += status_cusolver_;

    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  int LinSolverDirectCuSolverGLU::solve(vector_type* rhs, vector_type* x)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    status_cusolver_ = cusolverSpDgluSolve(handle_cusolversp_,
                                           A_->getNumRows(),
                                           /** A is original matrix */
                                           A_->getNnz(),
                                           descr_A_,
                                           A_->getValues(memory::DEVICE),
                                           A_->getRowData(memory::DEVICE),
                                           A_->getColData(memory::DEVICE),
                                           rhs->getData(memory::DEVICE), /** right hand side */
                                           x->getData(memory::DEVICE),   /** left hand side */
                                           &ite_refine_succ_,
                                           &r_nrm_inf_,
                                           info_M_,
                                           glu_buffer_);
    x->setDataUpdated(memory::DEVICE);
    RESOLVE_RANGE_POP(__FUNCTION__);
    return status_cusolver_;
  }

  int LinSolverDirectCuSolverGLU::solve(vector_type*)
  {
    out::error() << "Function solve(Vector* x) not implemented in CuSolverGLU!\n"
                 << "Consider using solve(Vector* rhs, Vector* x) instead.\n";
    return 1;
  }

  /***
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to set.
   * @return int Error code.
   */
  int LinSolverDirectCuSolverGLU::setCliParam(const std::string id, const std::string /** value */)
  {
    switch (getParamId(id))
    {
    default:
      std::cout << "Setting parameter failed!\n";
    }
    return 0;
  }

  /***
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return std::string Value of the string parameter to return.
   */
  std::string LinSolverDirectCuSolverGLU::getCliParamString(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown string parameter " << id << "\n";
    }
    return "";
  }

  /***
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return int Value of the int parameter to return.
   */
  index_type LinSolverDirectCuSolverGLU::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown integer parameter " << id << "\n";
    }
    return -1;
  }

  /***
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return real_type Value of the real_type parameter to return.
   */
  real_type LinSolverDirectCuSolverGLU::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown real parameter " << id << "\n";
    }
    return std::numeric_limits<real_type>::quiet_NaN();
  }

  /***
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return bool Value of the bool parameter to return.
   */
  bool LinSolverDirectCuSolverGLU::getCliParamBool(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    }
    return false;
  }

  int LinSolverDirectCuSolverGLU::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to print unknown parameter " << id << "\n";
      return 1;
    }
    return 0;
  }

} // namespace ReSolve
