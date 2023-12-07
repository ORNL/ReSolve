void mass_inner_product_two_vectors(int n, 
                                    int i, 
                                    double* vec1, 
                                    double* vec2, 
                                    double* mvec, 
                                    double* result);
void mass_axpy(int n, int i, double* x, double* y, double* alpha);

//needed for matrix inf nrm
void matrix_row_sums(int n, 
                     int nnz, 
                     int* a_ia,
                     double* a_val, 
                     double* result);

// needed for triangular solve

void permuteVectorP(int n, 
                    int* perm_vector,
                    double* vec_in, 
                    double* vec_out);
void permuteVectorQ(int n, 
                    int* perm_vector,
                    double* vec_in, 
                    double* vec_out);
void vector_inf_norm(int n,  
                     double* input,
                     double * buffer, 
                     double* result);
