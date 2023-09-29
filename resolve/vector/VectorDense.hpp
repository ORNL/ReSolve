#pragma once
#include <string>
#include <resolve/Common.hpp>
#include <resolve/vector/VectorBase.hpp>

namespace ReSolve { namespace vector {
  class VectorDense : public VectorBase
  {
    public:
      VectorDense(index_type n);
      ~VectorDense();

      void setDataUpdated(std::string memspace);
      int update(real_type* data, std::string memspaceIn, std::string memspaceOut);
      real_type* getData(std::string memspace);

      void allocate(std::string memspace);   

      void setData(real_type* data, std::string memspace);
      void setToZero(std::string memspace);
      void setToConst(real_type C, std::string memspace);
      int copyData(std::string memspaceIn, std::string memspaceOut); 
      real_type* getVectorData(std::string memspace); // get ith vector data out of multivector   
      int  deepCopyVectorData(real_type* dest, std::string memspace);  
      virtual int_type getCurrentVector() {return 0;} // this always returns 0;

    protected:
      index_type vector_current_{0};// we dont need it here but will need for multivectors to make the transition seamless
      index_type size_alloc_{n_}; // and same here
      real_type* d_data_{nullptr};
      real_type* h_data_{nullptr};

      // and this is to make life easier for vectors/multivectors and avoid repeated code
      int vecCopy(std::string memspaceIn, std::string memspaceOut, index_type start, index_type size);
      int vecCopyOut(std::string memspace, index_type start, index_type size, real_type* dataOut);
      int vecUpdate(std::string memspaceIn, std::string memspaceOut, index_type start, index_type size, real_type* dataIn);
      void vecZero(std::string memespace, index_type start, index_type size);
      void vecConst(std::string memespace, index_type start, index_type size);
      real_type* vecGet(std::string memspace, index_type start);

      // we keep info about EVERY VECTOR UPDATE
      bool* gpu_updated_;
      bool* cpu_updated_;
  };
}} // namespace ReSolve::vector
