#pragma once
namespace ReSolve::Hykkt
{
  enum PermutationType { PERM_V, REV_PERM_V, PERM_HES_V, PERM_JAC_V, PERM_JAC_TR_V };

  class PermClass
  {
  public:
  
    // constructor
    PermClass(int n_hes, int nnz_hes, int nnz_jac);
    
    // destructor
    ~PermClass();

    /*
    * @brief loads CSR structure for matrix H
    *
    * @param hes_i - Row offsets for H
    * hes_j - Column indices for H
    *
    * @post hes_i_ set to hes_i, hes_j_ set to hes_j
    */
    void addHInfo(int* hes_i, int* hes_j);

    /*
    * @brief loads CSR structure for matrix J
    *
    * @param jac_i - Row offsets for J
    * jac_j - Column indices for j
    * n_jac, m_jac - dimensions of J
    *
    * @post jac_i_ set to jac_i, jac_j_ set to jac_j, n_jac_ set to n_jac, m_jac_ set to m_jac
    */
    void addJInfo(int* jac_i, int* jac_j, int n_jac, int m_jac);

    /*
    * @brief loads CSR structure for matrix Jt
    *
    * @param jac_tr_i - Row offsets for Jt
    * jac_tr_j - Column indices for Jt
    *
    * @pre
    * @post jac_tr_i_ set to jac_tr_i, jac_tr_j_ set to jac_tr_j
    */
    void addJtInfo(int* jac_tr_i, int* jac_tr_j);

    /*
    * @brief sets custom permutation of matrix
    *
    * @param custom_perm - custom permutation vector
    *
    * @pre Member variable n_hes_ initialized to dimension of matrix
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
    * @pre Member variables n_hes_, nnz_hes_, hes_i_, hes_j_ have been 
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
    * @pre Member variables n_hes_, perm intialized to dimension of matrix
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
    * @param rhs_i - row offsets of permutation
    * rhs_j - column indices of permutation
    *
    * @pre Member variables n_hes_, nnz_hes_, hes_i_, hes_j_, perm, rev_perm
    *      initialized to the dimension of matrix H, number of nonzeros
    *      in H, row offsets for H, column indices for H, permutation
    *      and reverse permutation of H
    * 
    * @post perm_map_h is now permuted rows/columns of H and copied onto
    *       the device d_perm_map_h
    */
    void vecMapRC(int* rhs_i, int* rhs_j);

    /*
    * @brief Creates the permutation of the columns of matrix J
    * and copies onto device
    *
    * @param rhs_j - column indices of permutation
    *
    * @pre Member variables n_jac_, nnz_jac_, jac_i_, jac_j_, rev_perm initialized
    *      to the dimension of matrix J, number of nonzeros in J, row
    *      offsets for J, column indices for J, and reverse permutation
    * 
    * @post perm_map_jac is now the column permutation and is copied onto
    *       the device d_perm_map_jac
    */
    void vecMapC(int* rhs_j);

    /*
    * @brief Creates the permutation of the rows of matrix Jt
    * and copies onto device
    *
    * @param rhs_i - row offsets of permutation
    * rhs_j - column indices of permutation
    *
    * @pre Member variables m_jac_, nnz_jac_, jac_tr_i_, jac_tr_j_, initialized to
    *      the dimension of matrix J, the number of nonzeros in J, the
    *      row offsets for J transpose, the column indices for J transpose
    * 
    * @post perm_map_jac_tr is now the permuations of the rows of J transpose
    *       and is copied onto the device d_perm_map_jac_tr
    */
    void vecMapR(int* rhs_i, int* rhs_j);

    /*
    * @brief maps the permutated values of old_val to new_val
    *
    * @param permutation - the type of permutation of the 2x2 system
    * old_val - the old values in the matrix
    * new_val - the permuted values
    *
    * @pre Member variables n_hes_, nnz_hes_, nnz_jac_, d_perm, d_rev_perm,
    *      d_perm_map_h, d_perm_map_jac, d_perm_map_jac_tr initialized to
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
  * @pre Member variables n_hes_, nnz_hes_, nnz_jac_ are initialized to the
  *      dimension of matrix H, number of nonzeros in H, and number of
  *      nonzeros in matrix J
  *
  * @post perm_ and rev_perm_ are now vectors with size n_hes_, perm_map_h
  *       is now a vector with size nnz_hes_, perm_map_jac and perm_map_jac_tr
  *       are now vectors with size nnz_jac_
  */
    void allocateWorkspace();

    // member variables
    bool perm_is_default_ = true; // boolean if perm set custom
    
    int n_hes_; // dimension of H
    int nnz_hes_; // nonzeros of H

    int n_jac_; // dimensions of J
    int m_jac_;
    int nnz_jac_; // nonzeros of J

    int* perm_; // permutation of 2x2 system
    int* rev_perm_; // reverse of permutation
    int* perm_map_hes_; // mapping of permuted H
    int* perm_map_jac_; // mapping of permuted J
    int* perm_map_jac_tr_; // mapping of permuted Jt
    
    int* hes_i_; // row offsets of csr storage of H
    int* hes_j_; // column pointers of csr storage of H

    int* jac_i_; // row offsets of csr storage of J
    int* jac_j_; // column pointers of csr storage of J

    int* jac_tr_i_; // row offsets of csr storage of J transpose
    int* jac_tr_j_; // column pointers of csr storage of J transpose 

    // right hand side of 2x2 system
    double* rhs1_; // first block in vector
    double* rhs2_; // second block in vector
  };
}

