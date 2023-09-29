#pragma once
#include <string>
#include <resolve/Common.hpp>
#include <resolve/vector/VectorBase.hpp>
#include <resolve/vector/VectorDense.hpp>

namespace ReSolve { namespace vector {
  class VectorMulti : public VectorDense
  {
    public:
      VectorMulti(index_type n, index_type k);
      ~VectorMulti();

      void setCurrentVector(index_type i);
      index_type getCurrentVector();

      void setAllDataUpdated(std::string memspace);
      int updateAll(real_type* data, std::string memspaceIn, std::string memspaceOut);
      void setAllToZero(std::string memspace);
      void setAllToConst(std::string memspace);
      void deepCopyAllVectorData(real_type* dest, std::string memspace);
      void copyAllData(std::string memspaceIn, std::string memspaceOut);

    private:
      index_type k_; //num multivectors
  };
}
}

