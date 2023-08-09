#include <fstream>
#include <resolve/Vector.hpp>

// #include <resolve/matrix/Sparse.hpp>
// #include <resolve/matrix/Coo.hpp>

namespace ReSolve { namespace matrix { 
class Sparse;
class Coo;

namespace io {

  Coo* readMatrixFromFile(std::istream& file);
  void readAndUpdateMatrix(std::istream& file, Coo* A);
  real_type* readRhsFromFile(std::istream& file); 
  void readAndUpdateRhs(std::istream& file, real_type** rhs); 

  int writeMatrixToFile(Sparse* A, std::ostream file_out);
  int writeVectorToFile(Vector* vec_x, std::ostream file_out);

} // namespace io

}} // namespace ReSolve::matrix