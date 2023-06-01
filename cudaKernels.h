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
