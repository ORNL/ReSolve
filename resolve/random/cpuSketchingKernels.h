#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace cpu
  {
    void  count_sketch_theta(index_type n,
                            index_type k,
                            index_type* labels,
                            index_type* flip,
                            real_type* input,
                            real_type* output);

    void FWHT_scaleByD(index_type n,
                      const index_type* D,
                      const real_type* x,
                      real_type* y);

    void FWHT_select(index_type k,
                    const index_type* perm,
                    const real_type* input,
                    real_type* output);
    void FWHT(index_type M, index_type log2N, real_type* d_Data);
  }
}

