#!/bin/bash
cd ../ && make trips_py_test.exe
cd trips_code
../trips_py_test.exe input_files/trips_1d_deblur_A.mtx input_files/trips_1d_deblur_b_vec.mtx input_files/trips_1d_deblur_A.mtx
mv AB_output.mtx output_files
mv BA_output.mtx output_files
mv AB_reg_output.mtx output_files
mv Hybrid_AB_reg_output.mtx output_files

# ../trips_py_test.exe input_files/trips_2d_large_A.mtx input_files/trips_2d_large_b_vec.mtx input_files/trips_2d_large_A.mtx
# trips_code/input_files/trips_2d_large_A_24.mtx trips_code/input_files/trips_2d_large_b_vec_24.mtx trips_code/input_files/trips_2d_large_A_24.mtx