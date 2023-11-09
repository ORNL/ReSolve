#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>

namespace ReSolve
{
  LinAlgWorkspaceHIP::LinAlgWorkspaceHIP()
  {
    handle_rocsparse_   = nullptr;
    handle_rocblas_     = nullptr;

    matvec_setup_done_ = false;
  }

  LinAlgWorkspaceHIP::~LinAlgWorkspaceHIP()
  {
    rocsparse_destroy_handle(handle_rocsparse_);
    rocblas_destroy_handle(handle_rocblas_);
    rocsparse_destroy_mat_descr(mat_A_);
  }

  rocsparse_handle LinAlgWorkspaceHIP::getRocsparseHandle()
  {
    return handle_rocsparse_;
  }

  void LinAlgWorkspaceHIP::setRocsparseHandle(rocsparse_handle handle)
  {
    handle_rocsparse_ = handle;
  }

  rocblas_handle LinAlgWorkspaceHIP::getRocblasHandle()
  {
    return handle_rocblas_;
  }

  void LinAlgWorkspaceHIP::setRocblasHandle(rocblas_handle handle)
  {
    handle_rocblas_ = handle;
  }

  rocsparse_mat_descr LinAlgWorkspaceHIP::getSpmvMatrixDescriptor()
  {
    return mat_A_;
  }

  void LinAlgWorkspaceHIP::setSpmvMatrixDescriptor(rocsparse_mat_descr mat)
  {
    mat_A_ = mat;
  }

  rocsparse_mat_info  LinAlgWorkspaceHIP::getSpmvMatrixInfo()
  {
    return info_A_;
  }

  void LinAlgWorkspaceHIP::setSpmvMatrixInfo(rocsparse_mat_info  info)
  {
    info_A_ = info;
  }

  bool LinAlgWorkspaceHIP::matvecSetup()
  {
    return matvec_setup_done_;
  }

  void LinAlgWorkspaceHIP::matvecSetupDone()
  {
    matvec_setup_done_ = true;
  }

  void LinAlgWorkspaceHIP::initializeHandles()
  {
    rocsparse_create_handle(&handle_rocsparse_);
    rocblas_create_handle(&handle_rocblas_);
  }
} // namespace ReSolve
