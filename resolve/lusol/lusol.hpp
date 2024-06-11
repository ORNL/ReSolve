#pragma once

#include <resolve/Common.hpp>

// TODO: should we attach documentation comments to these or is there no point?

extern "C" {
  void lu1fac(ReSolve::index_type* m,
              ReSolve::index_type* n,
              ReSolve::index_type* nelem,
              ReSolve::index_type* lena,
              ReSolve::index_type* luparm,
              ReSolve::real_type* parmlu,
              ReSolve::real_type* a,
              ReSolve::index_type* indc,
              ReSolve::index_type* indr,
              ReSolve::index_type* p,
              ReSolve::index_type* q,
              ReSolve::index_type* lenc,
              ReSolve::index_type* lenr,
              ReSolve::index_type* locc,
              ReSolve::index_type* locr,
              ReSolve::index_type* iploc,
              ReSolve::index_type* iqloc,
              ReSolve::index_type* ipinv,
              ReSolve::index_type* iqinv,
              ReSolve::real_type* w,
              ReSolve::index_type* inform);

  void lu6sol(ReSolve::index_type* mode,
              ReSolve::index_type* m,
              ReSolve::index_type* n,
              ReSolve::real_type* v,
              ReSolve::real_type* w,
              ReSolve::index_type* lena,
              ReSolve::index_type* luparm,
              ReSolve::real_type* parmlu,
              ReSolve::real_type* a,
              ReSolve::index_type* indc,
              ReSolve::index_type* indr,
              ReSolve::index_type* p,
              ReSolve::index_type* q,
              ReSolve::index_type* lenc,
              ReSolve::index_type* lenr,
              ReSolve::index_type* locc,
              ReSolve::index_type* locr,
              ReSolve::index_type* inform);
}
