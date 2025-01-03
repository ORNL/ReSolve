#pragma once


enum PermutationType { perm_v, rev_perm_v, perm_h_v, perm_j_v, perm_jt_v }; 

class PermClass
{
public:
 
  // constructor
  PermClass(int n_h, int nnz_h, int nnz_j);
  
  // destructor
  ~PermClass();

  /*
   * @brief loads CSR structure for matrix H
   *
   * @param h_i - Row offsets for H
   * h_j - Column indices for H
   *
   * @post h_i_ set to h_i, h_j_ set to h_j
  */
  void addHInfo(int* h_i, int* h_j);

  /*
   * @brief loads CSR structure for matrix J
   *
   * @param j_i - Row offsets for J
   * j_j - Column indices for j
   * n_j, m_j - dimensions of J
   *
   * @post j_i_ set to j_i, j_j_ set to j_j, n_j_ set to n_j, m_j_ set to m_j
  */
  void addJInfo(int* j_i, int* j_j, int n_j, int m_j);

  /*
   * @brief loads CSR structure for matrix Jt
   *
   * @param jt_i - Row offsets for Jt
   * jt_j - Column indices for Jt
   *
   * @pre
   * @post jt_i_ set to jt_i, jt_j_ set to jt_j
  */
  void addJtInfo(int* jt_i, int* jt_j);

  /*
   * @brief sets custom permutation of matrix
   *
   * @param custom_perm - custom permutation vector
   *
   * @pre Member variable n_h_ initialized to dimension of matrix
   *
   * @post perm points to custom_perm out of scope so perm_is_default
   *       set to false so that custom_perm not deleted twice in destructors,
   *       permutation vector copied onto device d_perm
  */
  void addPerm(int* custom_perm);
  
  /*
   * @brief Uses Symmetric Approximate Minimum Degree 
   *        to reduce zero-fill in Cholesky Factorization
   *
   * @pre Member variables n_h_, nnz_h_, h_i_, h_j_ have been 
   *      initialized to the dimensions of matrix H, the number 
   *      of nonzeros it has, its row offsets, and column arrays
   *
   * @post perm is the perumation vector that implements symAmd
   *       on the 2x2 system
  */
  void symAmd();

  /*
   * @brief Creates reverse permutation of perm and copies onto device
   *
   * @pre Member variables n_h_, perm intialized to dimension of matrix
   *      and to a permutation vector
   * 
   * @post rev_perm is now the reverse permuation of perm and copied onto
   *       the device d_perm
  */
  void invertPerm();

  /*
   * @brief Creates permutation of rows and columns of matrix
   * and copies onto device
   *
   * @param b_i - row offsets of permutation
   * b_j - column indices of permutation
   *
   * @pre Member variables n_h_, nnz_h_, h_i_, h_j_, perm, rev_perm
   *      initialized to the dimension of matrix H, number of nonzeros
   *      in H, row offsets for H, column indices for H, permutation
   *      and reverse permutation of H
   * 
   * @post perm_map_h is now permuted rows/columns of H and copied onto
   *       the device d_perm_map_h
  */
  void vecMapRC(int* b_i, int* b_j);

  /*
   * @brief Creates the permutation of the columns of matrix J
   * and copies onto device
   *
   * @param b_j - column indices of permutation
   *
   * @pre Member variables n_j_, nnz_j_, j_i_, j_j_, rev_perm initialized
   *      to the dimension of matrix J, number of nonzeros in J, row
   *      offsets for J, column indices for J, and reverse permutation
   * 
   * @post perm_map_j is now the column permutation and is copied onto
   *       the device d_perm_map_j
  */
  void vecMapC(int* b_j);

  /*
   * @brief Creates the permutation of the rows of matrix Jt
   * and copies onto device
   *
   * @param b_i - row offsets of permutation
   * b_j - column indices of permutation
   *
   * @pre Member variables m_j_, nnz_j_, jt_i_, jt_j_, initialized to
   *      the dimension of matrix J, the number of nonzeros in J, the
   *      row offsets for J transpose, the column indices for J transpose
   * 
   * @post perm_map_jt is now the permuations of the rows of J transpose
   *       and is copied onto the device d_perm_map_jt
  */
  void vecMapR(int* b_i, int* b_j);

  /*
   * @brief maps the permutated values of old_val to new_val
   *
   * @param permutation - the type of permutation of the 2x2 system
   * old_val - the old values in the matrix
   * new_val - the permuted values
   *
   * @pre Member variables n_h_, nnz_h_, nnz_j_, d_perm, d_rev_perm,
   *      d_perm_map_h, d_perm_map_j, d_perm_map_jt initialized to
   *      the dimension of matrix H, number of nonzeros in H, number
   *      of nonzeros in matrix J, the device permutation and reverse
   *      permutation vectors, the device permutation mappings for 
   *      H, J, and J transpose
   * 
   * @post new_val contains the permuted old_val
  */
  void map_index(PermutationType permutation, 
      double* old_val, 
      double* new_val);

  void display_perm() const;

private:

/*
 * @brief allocates memory on host for permutation vectors
 *
 * @pre Member variables n_h_, nnz_h_, nnz_j_ are initialized to the
 *      dimension of matrix H, number of nonzeros in H, and number of
 *      nonzeros in matrix J
 *
 * @post perm_ and rev_perm_ are now vectors with size n_h_, perm_map_h
 *       is now a vector with size nnz_h_, perm_map_j and perm_map_jt
 *       are now vectors with size nnz_j_
*/
  void allocateWorkspace();

  // member variables
  bool perm_is_default_ = true; // boolean if perm set custom
  
  int n_h_; // dimension of H
  int nnz_h_; // nonzeros of H

  int n_j_; // dimensions of J
  int m_j_;
  int nnz_j_; // nonzeros of J

  int* perm_; // permutation of 2x2 system
  int* rev_perm_; // reverse of permutation
  int* perm_map_h_; // mapping of permuted H
  int* perm_map_j_; // mapping of permuted J
  int* perm_map_jt_; // mapping of permuted Jt
  
  int* h_i_; // row offsets of csr storage of H
  int* h_j_; // column pointers of csr storage of H

  int* j_i_; // row offsets of csr storage of J
  int* j_j_; // column pointers of csr storage of J

  int* jt_i_; // row offsets of csr storage of J transpose
  int* jt_j_; // column pointers of csr storage of J transpose 

  // right hand side of 2x2 system
  double* rhs1_; // first block in vector
  double* rhs2_; // second block in vector
};

