#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>

namespace ReSolve
{
  LinAlgWorkspaceHIP::LinAlgWorkspaceHIP()
  {
    handle_rocsparse_   = nullptr;
    handle_rocblas_     = nullptr;

    matvec_setup_done_ = false;
    d_r_               = nullptr;
    d_r_size_          = 0; 
    norm_buffer_       = nullptr;
  }

  LinAlgWorkspaceHIP::~LinAlgWorkspaceHIP()
  {
    rocsparse_destroy_handle(handle_rocsparse_);
    rocblas_destroy_handle(handle_rocblas_);
    rocsparse_destroy_mat_descr(mat_A_);
    if (d_r_size_ != 0)  mem_.deleteOnDevice(d_r_);
    if (norm_buffer_ != nullptr)  mem_.deleteOnDevice(d_r_);
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

  void LinAlgWorkspaceHIP::setDrSize(index_type new_sz)
  {
    d_r_size_ = new_sz;
  }
  
  void LinAlgWorkspaceHIP::setDr(double* new_dr)
  {
    d_r_ = new_dr;
  }
  
  void LinAlgWorkspaceHIP::setNormBuffer(double* nb)
  {
    norm_buffer_ = nb;
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
  
  index_type  LinAlgWorkspaceHIP::getDrSize()
  {
    return d_r_size_;
  }

  real_type*  LinAlgWorkspaceHIP::getDr()
  {
    return d_r_;
  }
  
  real_type*  LinAlgWorkspaceHIP::getNormBuffer()
  {
    return norm_buffer_;
  }
} // namespace ReSolve
