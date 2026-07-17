'''
Split HyKKT matrix into matrix blocks (D_d, D_s, J, J_d) and RHS vector into vector blocks (r_x, r_s, r_y, r_yd).

Usage: python ./hykkt_matrix_splitter.py [HyKKT matrix local path/URL] [HyKKT RHS local path/URL] [output directory]

Notes:
Matrix and RHS must be .mtx files if using local path. They can be .mtx.gz files if using URLs.
It doesn't matter if output directory has "/" at the end.

Might not work for very specific edge cases, such as when there is a very long diagonal of -1's outside the -I block.
'''

import sys
import os
import requests
import numpy as np
import scipy.io

if len(sys.argv) == 1:
    print("Arguments: matrix path/URL, RHS path/URL, output directory. Files can be .mtx or .mtx.gz.")
    sys.exit()

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    cp.cuda.Device(0).use()
    cp.cuda.runtime.setDevice(0)
except Exception:
    import scipy.sparse as cpsparse
    cp = np

output_dir = sys.argv[3].rstrip("/")

if sys.argv[1].startswith("http"):
    print("Downloading matrix file.")
    matrix_url = sys.argv[1]
    matrix_file = os.path.basename(matrix_url)
    if (matrix_file.endswith(".mtx.gz")):
        matrix_name = matrix_file[:-7]
    else:
        matrix_name = os.path.splitext(matrix_file)[0]
    matrix_path = f"{output_dir}/{matrix_file}"
    with open(matrix_path, "wb") as file:
        file.write(requests.get(matrix_url).content)
else:
    matrix_path = sys.argv[1]
    matrix_file = os.path.basename(matrix_path)
    matrix_name = os.path.splitext(matrix_file)[0]

if sys.argv[2].startswith("http"):
    print("Downloading rhs file.")
    rhs_url = sys.argv[2]
    rhs_file = os.path.basename(rhs_url)
    if rhs_file.endswith(".mtx.gz"):
        rhs_name = rhs_file[:-7]
    else:
        rhs_name = os.path.splitext(rhs_file)[0]
    rhs_path = f"{output_dir}/{rhs_file}"
    with open(rhs_path, "wb") as file:
        file.write(requests.get(rhs_url).content)
else:
    rhs_path = sys.argv[2]
    rhs_file = os.path.basename(rhs_path)
    rhs_name = os.path.splitext(rhs_file)[0]

print("Splitting matrix.")

A_cpu = scipy.io.mmread(matrix_path).tocoo()
n = A_cpu.shape[0]

A_row = cp.asarray(A_cpu.row)
A_col = cp.asarray(A_cpu.col)
A_val = cp.asarray(A_cpu.data)

mo_mask = (A_val == -1)
mo_row = A_row[mo_mask]
mo_col = A_col[mo_mask]

offsets = mo_row - mo_col
mI_mask = (offsets > 0)
offsets = offsets[mI_mask]
assert len(offsets) > 0, "Not a valid HyKKT matrix!"

unique_offsets, counts = cp.unique(offsets, return_counts=True)
most_common_offset = unique_offsets[cp.argmax(counts)]

mI_block_col = mo_col[mI_mask]
mI_block_col = cp.sort(mI_block_col[offsets == most_common_offset])
assert mI_block_col[-1] - mI_block_col[0] == len(mI_block_col) - 1, "Stray -1!"

nx = mI_block_col[0]
md = len(mI_block_col)
mc = (n - nx - md - md).item()

print(f"n_x={nx}, m_d={md}, m_c={mc}")

R0_mask = A_row < nx
R1_mask = ~R0_mask & (A_row < nx + md)
R2_mask = ~R0_mask & ~R1_mask & (A_row < nx + md + mc)
R3_mask = ~R0_mask & ~R1_mask & ~R2_mask

C0_mask = A_col < nx
C1_mask = ~C0_mask & (A_col < nx + md)
C2_mask = ~C0_mask & ~C1_mask & (A_col < nx + md + mc)
C3_mask = ~C0_mask & ~C1_mask & ~C2_mask

H_mask  = R0_mask & C0_mask
Dd_mask = R1_mask & C1_mask
J_mask  = R2_mask & C0_mask
Jd_mask = R3_mask & C0_mask

R0_end = nx
R1_end = nx + md
R2_end = nx + md + mc
C0_end = nx

H_block  = cpsparse.coo_matrix((A_val[H_mask], (A_row[H_mask], A_col[H_mask])), shape=(nx, nx))
Dd_block = cpsparse.coo_matrix((A_val[Dd_mask], (A_row[Dd_mask] - R0_end, A_col[Dd_mask] - C0_end)), shape=(md, md))
J_block  = cpsparse.coo_matrix((A_val[J_mask], (A_row[J_mask] - R1_end, A_col[J_mask])), shape=(mc, nx))
Jd_block = cpsparse.coo_matrix((A_val[Jd_mask], (A_row[Jd_mask] - R2_end, A_col[Jd_mask])), shape=(md, nx))

print(f"H nnz={H_block.nnz}, D_s nnz={Dd_block.nnz}, J nnz={J_block.nnz}, Jd nnz={Jd_block.nnz}")

if cp == np:
    scipy.io.mmwrite(f"{output_dir}/block_H_{matrix_name}.mtx", H_block)
    scipy.io.mmwrite(f"{output_dir}/block_Dd_{matrix_name}.mtx", Dd_block)
    scipy.io.mmwrite(f"{output_dir}/block_J_{matrix_name}.mtx", J_block)
    scipy.io.mmwrite(f"{output_dir}/block_Jd_{matrix_name}.mtx", Jd_block)
else:
    scipy.io.mmwrite(f"{output_dir}/block_H_{matrix_name}.mtx", H_block.get())
    scipy.io.mmwrite(f"{output_dir}/block_Dd_{matrix_name}.mtx", Dd_block.get())
    scipy.io.mmwrite(f"{output_dir}/block_J_{matrix_name}.mtx", J_block.get())
    scipy.io.mmwrite(f"{output_dir}/block_Jd_{matrix_name}.mtx", Jd_block.get())

rhs = cp.asarray(scipy.io.mmread(rhs_path)).flatten()

rx = rhs[: R0_end]
rs = rhs[R0_end : R1_end]
ry = rhs[R1_end : R2_end]
ryd = rhs[R2_end :]

if cp == np:
    scipy.io.mmwrite(f"{output_dir}/block_rx_{rhs_name}.mtx", rx.reshape(-1, 1))
    scipy.io.mmwrite(f"{output_dir}/block_rs_{rhs_name}.mtx", rs.reshape(-1, 1))
    scipy.io.mmwrite(f"{output_dir}/block_ry_{rhs_name}.mtx", ry.reshape(-1, 1))
    scipy.io.mmwrite(f"{output_dir}/block_ryd_{rhs_name}.mtx", ryd.reshape(-1, 1))
else:
    scipy.io.mmwrite(f"{output_dir}/block_rx_{rhs_name}.mtx", rx.get().reshape(-1, 1))
    scipy.io.mmwrite(f"{output_dir}/block_rs_{rhs_name}.mtx", rs.get().reshape(-1, 1))
    scipy.io.mmwrite(f"{output_dir}/block_ry_{rhs_name}.mtx", ry.get().reshape(-1, 1))
    scipy.io.mmwrite(f"{output_dir}/block_ryd_{rhs_name}.mtx", ryd.get().reshape(-1, 1))

if sys.argv[1].startswith("http"):
    os.remove(matrix_path)
if sys.argv[2].startswith("http"):
    os.remove(rhs_path)

print("Done.")
