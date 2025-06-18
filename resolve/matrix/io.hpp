#include <fstream>

namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }
} // namespace ReSolve

namespace ReSolve
{
  namespace matrix
  {
    class Sparse;
    class Coo;
    class Csr;
  } // namespace matrix
} // namespace ReSolve

namespace ReSolve
{
  namespace io
  {
    using vector_type = vector::Vector;

    matrix::Coo*    createCooFromFile(std::istream& file, bool is_expand_symmetric = true);
    matrix::Csr*    createCsrFromFile(std::istream& file, bool is_expand_symmetric = true);
    void            updateMatrixFromFile(std::istream& file, matrix::Coo* A);
    void            updateMatrixFromFile(std::istream& file, matrix::Csr* A);
    real_type*      createArrayFromFile(std::istream& file);
    vector::Vector* createVectorFromFile(std::istream& file);
    void            updateArrayFromFile(std::istream& file, real_type** rhs);
    void            updateVectorFromFile(std::istream& file, vector::Vector* rhs);

    int writeMatrixToFile(matrix::Sparse* A, std::ostream& file_out);
    int writeVectorToFile(vector_type* vec_x, std::ostream& file_out);
  } // namespace io
} // namespace ReSolve
