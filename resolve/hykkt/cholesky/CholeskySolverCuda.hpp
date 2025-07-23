namespace ReSolve {
  namespace hykkt {
    class CholeskySolverCuda : public CholeskySolverImpl {
    public:
      CholeskySolverCuda() = default;
      ~CholeskySolverCuda() override = default;

      void symbolicAnalysis(matrix::Csr* A) override;
      void numericalFactorization(matrix::Csr* A, real_type tol) override;
      void solve(vector::Vector* x, vector::Vector* b) override;
    private:
      //handle to the cuSPARSE library context
    cusolverSpHandle_t handle_cusolver_;
    cusparseMatDescr_t descr_a_;//descriptor for matrix A 
    csrcholInfo_t info_; // stores Cholesky factorization
    void* buffer_; // buffer for Cholesky factorization
  }
}