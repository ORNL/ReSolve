#include <fstream>

namespace ReSolve { namespace vector { 
  class Vector;
}}

namespace ReSolve { namespace matrix { 
  class Sparse;
  class Coo;
}}

namespace ReSolve { namespace io {
  using vector_type = vector::Vector;

  matrix::Coo* readMatrixFromFile(std::istream& file, bool is_expand_symmetric = true);
  void readAndUpdateMatrix(std::istream& file, matrix::Coo* A);
  real_type* readRhsFromFile(std::istream& file); 
  void readAndUpdateRhs(std::istream& file, real_type** rhs); 

  int writeMatrixToFile(matrix::Sparse* A, std::ostream& file_out);
  int writeVectorToFile(vector_type* vec_x, std::ostream &file_out);
}} // ReSolve::io
