/**
 * @file runHykktCGTests.hpp
 * @brief Tests for class hykkt::SchurComplementConjugateGradient
 *
 */
#include <fstream>
#include <iostream>
#include <string>

#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif
#ifdef RESOLVE_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

#include "HykktCGTests.hpp"
#include <resolve/vector/Vector.hpp>

/**
 * @brief Run tests with a given backend
 *
 * @param backend - string name of the hardware backend
 * @param result - test results
 */
template <typename WorkspaceType>
void runTests(const std::string& backend, ReSolve::memory::MemorySpace memspace, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running conjugate gradient tests on " << backend << " device:\n";

  WorkspaceType workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler                                     matrix_handler(&workspace);
  ReSolve::VectorHandler                                     vector_handler(&workspace);
  ReSolve::tests::HykktConjugateGradientTests<WorkspaceType> test(memspace, matrix_handler, vector_handler, workspace);

  std::string        source_dir = std::string(SOURCE_DIR);
    
  std::vector<std::string> matrix_names{
    // "mhd4800b", // WARMUP
    // "mhd4800b", // WARMUP
    // "mhd4800b", // WARMUP
    "bcsstk18", // WARMUP

    "hood",
    "Fault_639",
    // "2cubes_sphere",

    // "1138_bus",
    // "Chem97ZtZ",
    // "sts4098",
    // "bcsstk13",
    // "bcsstk18",
    // "torsion1",
    // "shallow_water1",
    // "Kuu",
    // "cvxbqp1",
    // "gridgena",
    // "wathen120",
    // "finan512",
    // "thermomech_TC",
    // "Pres_Poisson",
    // "G2_circuit",
    // "bundle1",
    // "qa8fm",
    // "cfd2",
    "parabolic_fem",
    // "ecology2",
    // "tmt_sym",
    // "G3_circuit",

    // "cbuckle",
    // "BenElechi1",
    // "msdoor",
    // "apache1",
    // "x104",//
    // "thread", //
    // "shipsec8",
    // "ship_001",
    // "m_t1", //
    // "bmwcra_1",
    // "bmw7st_1",//
    // "af_shell4",
    "nd6k",
    // "olafu",
    // "consph",

    // "s3dkq4m2",
    // "vanbody",//
    // "raefsky4",
    // "nasasrb",
    // "ct20stif",
    // "PFlow_742",//

    // "cant",
    // "offshore",

    // "Dubcova3",
    // "obstclae",
    // "ship_003",
    // "nd24k",
    // "bcsstk38",
    // "bcsstk17",
    // "s2rmq4m1",

    // "pwtk",
    // "crankseg_2",
    // "Fault_639",
    // "ldoor",
    // "boneS10",
    // "Hook_1498",
    // "bone010",
    // "Flan_1565",

    // "msc23052",//
    // "msc10848",
    // "msc04515",
    // "bcsstk24",
    // "bcsstk15",
    // "Trefethen_20000",
    // "Dubcova1",
    // "ted_B",
    // "minsurfo",
    // "jnlbrng1",
    // "audikw_1",//
    // "t2dah_e",
    // "nd12k",
    // "nd3k",
    // "fv3",
    // "bodyy4",
    // "aft01",

    // "t3dl_e",
    // "nasa1824",
    // "msc01440",
    // "msc00726",
    // "bcsstm39",
    // "nos3",
    // "bcsstk11",
    // "bcsstk06",
    // "bcsstk04",
    // "Trefethen_700",
    // "mhd4800b",
    // "Journals",

    // "bcsstm26",
    // "bcsstm22",
    // "bcsstm20",
    // "bcsstm19",
    // "bcsstm11",
    // "bcsstm09",
    // "bcsstm08",
    // "bcsstm06",
    // "bcsstm05",
    // "bcsstm02",
    // "bcsstk22",
    // "bcsstk03",
    // "bcsstk01",
    // "494_bus",
    // "mesh3em5",
    // "mesh1em6",
    // "ex5",
    // "nos1",

    // "bcsstk34",
    // "bcsstk28",
    // "bcsstk25",
    // "bcsstk10",
    // "plbuckle",
    // "msc01050"
    // "thermomech_dM",
    // "oilpan",
    // "bundle_adj",
    // "Geo_1438",
  };

  for (const std::string& matrix_name : matrix_names)
  {
    std::string        A_file_name = source_dir + std::string("/MBPCGTestMatrices/") + matrix_name + std::string(".mtx");
    std::string        b_file_name = source_dir + std::string("/MBPCGTestMatrices/") + matrix_name + std::string("_b.mtx");

    printf("\n\n\nMatrix: %s\n", matrix_name.c_str());
    result += test.CGTest(A_file_name, b_file_name, false);

    std::cout << "\n";
  }
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;
  // runTests<ReSolve::LinAlgWorkspaceCpu>("CPU", ReSolve::memory::HOST, result);

#ifdef RESOLVE_USE_CUDA
  runTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests<ReSolve::LinAlgWorkspaceHIP>("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
