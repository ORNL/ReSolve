namespace ReSolve {
  namespace hykkt {
    class CholeskySolverImpl {
    public:
      CholeskySolverImpl();
      ~CholeskySolverImpl();

      void symbolicAnalysis(matrix::Csr* A);
      void numericalFactorization(matrix::Csr* A, real_type tol);
      void solve(vector::Vector* x, vector::Vector* b);
      
    };
  }
}