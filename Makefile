CC := g++
NVCC := nvcc
CUDA_LIBS := -lcusparse -lcublas -lcusolver
LIBS := -lklu
SRCCPU := MatrixHandler.cpp  VectorHandler.cpp MatrixIO.cpp LinSolver.cpp LinSolverDirectKLU.cpp r_KLU_rf_FGMRES.cpp
OBJSCPU := $(foreach f,$(SRCCPU),$(f:%.cpp=%.o))

SRCCUDA := Matrix.cpp Vector.cpp LinAlgWorkspace.cpp LinSolverDirectCuSolverRf.cpp LinSolverDirectCuSolverGLU.cpp LinSolverIterativeFGMRES.cpp

OBJSCUDA := $(foreach f,$(SRCCUDA),$(f:%.cpp=%.o))

SRCEXTRA:= cudaKernels.cu
OBJSEXTRA:=  $(foreach f,$(SRCEXTRA),$(f:%.cu=%.o))


all: ${OBJSCPU} ${OBJSCUDA} ReSolve

%.o: %.cpp
	${CC} ${CUDALIBS} -o $@ -c $<

%.o: %.cu
	${NVCC} ${CUDALIBS} -o $@ -c $<

ReSolve: ${OBJSCPU} ${OBJSGPU} ${OBJSEXTRA}
	${NVCC} -o $@ ${OBJSCPU} ${OBJSCUDA} ${OBJSEXTRA} ${LIBS} ${CUDA_LIBS}

clean:
	rm -f *.o
	rm ReSolve
