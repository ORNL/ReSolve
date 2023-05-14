#include "resolveMatrix.hpp"
#include <cuda_runtime.h>

namespace ReSolve {

	resolveMatrix::resolveMatrix()
	{
	}

	resolveMatrix::resolveMatrix(resolveInt n, 
															 resolveInt m, 
															 resolveInt nnz):
		n{n},
		m{m},
		nnz{nnz}
	{
		this->is_symmetric = false;
		this->is_expanded = true;//defaults is a normal non-symmetric fully expanded matrix
		this->nnz_expanded = nnz;
	}
	resolveMatrix::resolveMatrix(resolveInt n, 
															 resolveInt m, 
															 resolveInt nnz,
															 bool symmetric,
															 bool expanded):
		n{n},
		m{m},
		nnz{nnz},
		is_symmetric{symmetric},
		is_expanded{expanded}
	{
		if (is_expanded) {
			this->nnz_expanded = nnz;
		} else {
			this->nnz_expanded = 0;
		}
	}

	resolveMatrix::~resolveMatrix()
	{
	}

	resolveInt resolveMatrix::getNumRows()
	{
		return this->n;
	}

	resolveInt resolveMatrix::getNumColumns()
	{
		return this->m;
	}

	resolveInt resolveMatrix::getNnz()
	{
		return this->nnz;
	}

	resolveInt resolveMatrix::getNnzExpanded()
	{
		return this->nnz_expanded;
	}

	bool resolveMatrix::symmetric()
	{
		return is_symmetric;
	}

	bool resolveMatrix::expanded()
	{
		return is_expanded;
	}

	void resolveMatrix::setSymmetric(bool symmetric)
	{
		is_symmetric = symmetric;
	}

	void resolveMatrix::setExpanded(bool expanded)
	{
		is_expanded = expanded;
	}

	void resolveMatrix::setNnzExpanded(resolveInt nnz_expanded_new)
	{
		nnz_expanded = nnz_expanded_new;
	}

	resolveInt* resolveMatrix::getCsrRowPointers(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_csr_p;
		} else {
			if (memspace == "cuda") {
				return this->d_csr_p;
			} else {
				return nullptr;
			}
		}
	}

	resolveInt* resolveMatrix::getCsrColIndices(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_csr_i;
		} else {
			if (memspace == "cuda") {
				return this->d_csr_i;
			} else {
				return nullptr;
			}
		}
	}

	resolveReal* resolveMatrix::getCsrValues(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_csr_x;
		} else {
			if (memspace == "cuda") {
				return this->d_csr_x;
			} else {
				return nullptr;
			}
		}
	}

	resolveInt* resolveMatrix::getCscColPointers(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_csc_p;
		} else {
			if (memspace == "cuda") {
				return this->d_csc_p;
			} else {
				return nullptr;
			}
		}
	}

	resolveInt* resolveMatrix::getCscRowIndices(std::string memspace) 
	{
		if (memspace == "cpu") {
			return this->h_csc_i;
		} else {
			if (memspace == "cuda") {
				return this->d_csc_i;
			} else {
				return nullptr;
			}
		}
	}

	resolveReal* resolveMatrix::getCscValues(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_csc_x;
		} else {
			if (memspace == "cuda") {
				return this->d_csc_x;
			} else {
				return nullptr;
			}
		}
	}

	resolveInt* resolveMatrix::getCooRowIndices(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_coo_rows;
		} else {
			if (memspace == "cuda") {
				return this->d_coo_rows;
			} else {
				return nullptr;
			}
		}
	}

	resolveInt* resolveMatrix::getCooColIndices(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_coo_cols;
		} else {
			if (memspace == "cuda") {
				return this->d_coo_cols;
			} else {
				return nullptr;
			}
		}
	}

	resolveReal* resolveMatrix::getCooValues(std::string memspace)
	{
		if (memspace == "cpu") {
			return this->h_coo_vals;
		} else {
			if (memspace == "cuda") {
				return this->d_coo_vals;
			} else {
				return nullptr;
			}
		}
	}

	int resolveMatrix::setCsr(int* csr_p, int* csr_i, double* csr_x, std::string memspace)
	{
		if (memspace == "cpu"){
			this->h_csr_p = csr_p;
			this->h_csr_i = csr_i;
			this->h_csr_x = csr_x;
		} else {
			if (memspace == "cuda"){ 
				this->d_csr_p = csr_p;
				this->d_csr_i = csr_i;
				this->d_csr_x = csr_x;
			} else {
				return -1;
			}

		}
		return 0;
	}

	int resolveMatrix::setCsc(int* csc_p, int* csc_i, double* csc_x, std::string memspace)
	{
		if (memspace == "cpu"){
			this->h_csc_p = csc_p;
			this->h_csc_i = csc_i;
			this->h_csc_x = csc_x;
		} else {
			if (memspace == "cuda"){ 
				this->d_csc_p = csc_p;
				this->d_csc_i = csc_i;
				this->d_csc_x = csc_x;
			} else {
				return -1;
			}

		}
		return 0;
	}

	int resolveMatrix::setCoo(int* coo_rows, int* coo_cols, double* coo_vals, std::string memspace)
	{
		if (memspace == "cpu"){
			this->h_coo_rows = coo_rows;
			this->h_coo_cols = coo_cols;
			this->h_coo_vals = coo_vals;
		} else {
			if (memspace == "cuda"){ 
				this->d_coo_rows = coo_rows;
				this->d_coo_cols = coo_cols;
				this->d_coo_vals = coo_vals;
			} else {
				return -1;
			}
		}
		return 0;
	}


	resolveInt resolveMatrix::updateCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x,  std::string memspaceIn, std::string memspaceOut)
	{

		//four cases (for now)
		int control=-1;
		if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
		if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
		if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
		if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

		if (memspaceOut == "cpu") {
			//check if cpu data allocated
			if (h_csr_p == nullptr) {
				this->h_csr_p = new resolveInt[n+1];
			}
			if (h_csr_i == nullptr) {
				this->h_csr_i = new resolveInt[nnz];
			} 
			if (h_csr_x == nullptr) {
				this->h_csr_x = new resolveReal[nnz];
			}
		}

		if (memspaceOut == "cuda") {
			//check if cuda data allocated
			if (d_csr_p == nullptr) {
				cudaMalloc(&d_csr_p, (n + 1)*sizeof(resolveInt)); 
			}
			if (d_csr_i == nullptr) {
				cudaMalloc(&d_csr_i, nnz * sizeof(resolveInt)); 
			}
			if (d_csr_x == nullptr) {
				cudaMalloc(&d_csr_x, nnz * sizeof(resolveReal)); 
			}
		}

		//copy	
		switch(control)  {
			case 0: //cpu->cpu
				std::memcpy(h_csr_p, csr_p, (n + 1) * sizeof(resolveInt));
				std::memcpy(h_csr_i, csr_i, (nnz) * sizeof(resolveInt));
				std::memcpy(h_csr_x, csr_x, (nnz) * sizeof(resolveReal));
				break;
			case 1://cuda->cpu
				cudaMemcpy(h_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_csr_i, csr_i, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_csr_x, csr_x, (nnz) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
				break;
			case 2://cpu->cuda
				cudaMemcpy(d_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
				cudaMemcpy(d_csr_i, csr_i, (nnz) * sizeof(resolveInt), cudaMemcpyHostToDevice);
				cudaMemcpy(d_csr_x, csr_x, (nnz) * sizeof(resolveReal), cudaMemcpyHostToDevice);
				break;
			case 3://cuda->cuda
				cudaMemcpy(d_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
				cudaMemcpy(d_csr_i, csr_i, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
				cudaMemcpy(d_csr_x, csr_x, (nnz) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
				break;
			default:
				return -1;
		}
		return 0;
	}

	resolveInt resolveMatrix::updateCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x,  std::string memspaceIn, std::string memspaceOut)
	{

		//four cases (for now)
		int control=-1;
		if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
		if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
		if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
		if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

		if (memspaceOut == "cpu") {
			//check if cpu data allocated
			if (h_csc_p == nullptr) {
				this->h_csc_p = new resolveInt[n+1];
			}
			if (h_csc_i == nullptr) {
				this->h_csc_i = new resolveInt[nnz];
			} 
			if (h_csc_x == nullptr) {
				this->h_csc_x = new resolveReal[nnz];
			}
		}

		if (memspaceOut == "cuda") {
			//check if cuda data allocated
			if (d_csc_p == nullptr) {
				cudaMalloc(&d_csc_p, (n + 1)*sizeof(resolveInt)); 
			}
			if (d_csc_i == nullptr) {
				cudaMalloc(&d_csc_i, nnz * sizeof(resolveInt)); 
			}
			if (d_csc_x == nullptr) {
				cudaMalloc(&d_csc_x, nnz * sizeof(resolveReal)); 
			}
		}

		switch(control)  {
			case 0: //cpu->cpu
				std::memcpy(h_csc_p, csc_p, (n + 1) * sizeof(resolveInt));
				std::memcpy(h_csc_i, csc_i, (nnz) * sizeof(resolveInt));
				std::memcpy(h_csc_x, csc_x, (nnz) * sizeof(resolveReal));
				break;
			case 1://cuda->cpu
				cudaMemcpy(h_csc_p, csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_csc_i, csc_i, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_csc_x, csc_x, (nnz) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
				break;
			case 2://cpu->cuda
				cudaMemcpy(d_csc_p, csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
				cudaMemcpy(d_csc_i, csc_i, (nnz) * sizeof(resolveInt), cudaMemcpyHostToDevice);
				cudaMemcpy(d_csc_x, csc_x, (nnz) * sizeof(resolveReal), cudaMemcpyHostToDevice);
				break;
			case 3://cuda->cuda
				cudaMemcpy(d_csc_p, csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
				cudaMemcpy(d_csc_i, csc_i, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
				cudaMemcpy(d_csc_x, csc_x, (nnz) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
				break;
			default:
				return -1;
		}
		return 0;
	}

	resolveInt resolveMatrix::updateCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals,  std::string memspaceIn, std::string memspaceOut)
	{

		//four cases (for now)
		int control=-1;
		if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
		if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
		if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
		if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

		if (memspaceOut == "cpu") {
			//check if cpu data allocated	
			if (h_coo_rows == nullptr) {
				this->h_coo_rows = new resolveInt[nnz];
			}
			if (h_coo_cols == nullptr) {
				this->h_coo_cols = new resolveInt[nnz];
			}
			if (h_coo_vals == nullptr) {
				this->h_coo_vals = new resolveReal[nnz];
			}
		}

		if (memspaceOut == "cuda") {
			//check if cuda data allocated
			if (d_coo_rows == nullptr) {
				cudaMalloc(&d_coo_rows, (nnz) * sizeof(resolveInt)); 
			}
			if (d_coo_cols == nullptr) {
				cudaMalloc(&d_coo_cols, nnz * sizeof(resolveInt)); 
			}
			if (d_coo_vals == nullptr) {
				cudaMalloc(&d_coo_vals, nnz * sizeof(resolveReal)); 
			}
		}

		switch(control)  {
			case 0: //cpu->cpu
				std::memcpy(h_coo_rows, coo_rows, (nnz) * sizeof(resolveInt));
				std::memcpy(h_coo_cols, coo_cols, (nnz) * sizeof(resolveInt));
				std::memcpy(h_coo_vals, coo_vals, (nnz) * sizeof(resolveReal));
				break;
			case 1://cuda->cpu
				cudaMemcpy(h_coo_rows, coo_rows, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_coo_cols, coo_cols, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_coo_vals, coo_vals, (nnz) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
				break;
			case 2://cpu->cuda
				cudaMemcpy(d_coo_rows, coo_rows, (nnz) * sizeof(resolveInt), cudaMemcpyHostToDevice);
				cudaMemcpy(d_coo_cols, coo_cols, (nnz) * sizeof(resolveInt), cudaMemcpyHostToDevice);
				cudaMemcpy(d_coo_vals, coo_vals, (nnz) * sizeof(resolveReal), cudaMemcpyHostToDevice);
				break;
			case 3://cuda->cuda
				cudaMemcpy(d_coo_rows, coo_rows, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
				cudaMemcpy(d_coo_cols, coo_cols, (nnz) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
				cudaMemcpy(d_coo_vals, coo_vals, (nnz) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
				break;
			default:
				return -1;
		}
		return 0;
	}


	resolveInt resolveMatrix::updateCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut)
	{
		this->destroyCsr(memspaceOut);
		int i = this->updateCsc(csr_p, csr_i, csr_x, memspaceIn, memspaceOut);
return i;
  }
	
	resolveInt resolveMatrix::updateCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut)
	{
		this->destroyCsc(memspaceOut);
		int i = this->updateCsc(csc_p, csc_i, csc_x, memspaceIn, memspaceOut);
return i;
  }

	resolveInt resolveMatrix::updateCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut)
	{
		this->destroyCoo(memspaceOut);
		int i = this->updateCoo(coo_rows, coo_cols, coo_vals, memspaceIn, memspaceOut);
return i;
  }

	resolveInt resolveMatrix::destroyCsr(std::string memspace)
	{
		if (memspace == "cpu"){  
			if (h_csr_p != nullptr) delete [] h_csr_p;
			if (h_csr_i != nullptr) delete [] h_csr_i;
			if (h_csr_x != nullptr) delete [] h_csr_x;
		} else {
			if (memspace == "cuda"){ 
				if (d_csr_p != nullptr) cudaFree(h_csr_p);
				if (d_csr_i != nullptr) cudaFree(h_csr_i);
				if (d_csr_x != nullptr) cudaFree(h_csr_x);
			} else {
				return -1;
			}
		}
		return 0;
	}

	resolveInt resolveMatrix::destroyCsc(std::string memspace)
	{   
		if (memspace == "cpu"){  
			if (h_csc_p != nullptr) delete [] h_csc_p;
			if (h_csc_i != nullptr) delete [] h_csc_i;
			if (h_csc_x != nullptr) delete [] h_csc_x;
		} else {
			if (memspace == "cuda"){ 
				if (d_csc_p != nullptr) cudaFree(h_csc_p);
				if (d_csc_i != nullptr) cudaFree(h_csc_i);
				if (d_csc_x != nullptr) cudaFree(h_csc_x);
			} else {
				return -1;
			}
		}
		return 0;
	}

	resolveInt resolveMatrix::destroyCoo(std::string memspace)
	{ 
		if (memspace == "cpu"){  
			if (h_coo_rows != nullptr) delete [] h_coo_rows;
			if (h_coo_cols != nullptr) delete [] h_coo_cols;
			if (h_coo_vals != nullptr) delete [] h_coo_vals;
		} else {
			if (memspace == "cuda"){ 
				if (d_coo_rows != nullptr) cudaFree(h_coo_rows);
				if (d_coo_cols != nullptr) cudaFree(h_coo_cols);
				if (d_coo_vals != nullptr) cudaFree(h_coo_vals);
			} else {
				return -1;
			}
		}
		return 0;
	}
}
