#!/bin/bash

#./trips_py_test.exe trips_1d_deblur_A.mtx trips_1d_deblur_b_true.mtx trips_1d_deblur_A.mtx 
python3 ./trips_graph.py .././input_files/trips_1d_deblur_A.mtx .././output_files/AB_output.mtx .././input_files/trips_1d_deblur_x_true.mtx trips_AB_plot.pdf
python3 ./trips_graph.py .././input_files/trips_1d_deblur_A.mtx .././output_files/BA_output.mtx .././input_files/trips_1d_deblur_x_true.mtx trips_BA_plot.pdf
python3 ./trips_graph.py .././input_files/trips_1d_deblur_A.mtx .././output_files/AB_reg_output.mtx .././input_files/trips_1d_deblur_x_true.mtx trips_AB_reg_plot.pdf
python3 ./trips_graph.py .././input_files/trips_1d_deblur_A.mtx .././output_files/Hybrid_AB_reg_output.mtx .././input_files/trips_1d_deblur_x_true.mtx trips_HybridAB_reg_plot.pdf