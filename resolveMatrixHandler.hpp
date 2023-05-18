// this class encapsulates various matrix manipulation operations, commonly required by linear solvers:
// this includes 
// (1) Matrix format conversion: coo2csr, csr2csc
// (2) Matrix vector product (SpMV)
// (3) Matrix 1-norm
#include "resolveMatrix.hpp"
#include "resolveLinAlgWorkspace.hpp"
#include <algorithm>

namespace ReSolve
{
  //helper class
  class indexPlusValue
  {
    public:
      indexPlusValue();
      ~indexPlusValue();
      void setIdx (resolveInt new_idx);
      void setValue (resolveReal new_value);

      resolveInt getIdx();
      resolveReal getValue();

      bool operator < (const indexPlusValue& str) const
      {
        return (idx < str.idx);
      }  

    private:
      resolveInt idx;
      resolveReal value;

  };

  class resolveMatrixHandler
  {
    public:
      resolveMatrixHandler();
      resolveMatrixHandler(resolveLinAlgWorkspace* workspace);
      ~resolveMatrixHandler();

      void csr2csc(resolveMatrix* A, std::string memspace);//memspace decides on what is returned (cpu or cuda pointer)
      void coo2csr(resolveMatrix* A, std::string memspace);

      void resolveMatvec(resolveMatrix* A, resolveReal* x, resolveReal* result, resolveReal* alpha, resolveReal* beta, std::string memspace);
      void resolveMatrix1Norm(resolveMatrix *A, resolveReal* norm);

    private: 
      resolveLinAlgWorkspace* workspace;
      bool new_matrix; //if the structure changed, you need a new handler.
  };
}

