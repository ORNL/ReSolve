CC := g++
NVCC := nvcc
CUDA_LIBS := -lcusparse -lcublas -lcusolver
LIBS := -lklu
SRCCPU := MatrixHandler.cpp  VectorHandler.cpp MatrixIO.cpp LinSolver.cpp LinSolverDirectKLU.cpp main.cpp
OBJSCPU := $(foreach f,$(SRCCPU),$(f:%.cpp=%.o))

SRCCUDA := Matrix.cpp Vector.cpp LinAlgWorkspace.cpp LinSolverDirectCuSolverRf.cpp LinSolverDirectCuSolverGLU.cpp

OBJSCUDA := $(foreach f,$(SRCCUDA),$(f:%.cpp=%.o))

all: ${OBJSCPU} ${OBJSCUDA} ReSolve

%.o: %.cpp
	${CC} ${CUDALIBS} -o $@ -c $<
ReSolve: ${OBJCPU} ${OBJGPU}
	${NVCC} -o $@ ${OBJSCPU} ${OBJSCUDA} ${LIBS} ${CUDA_LIBS}

clean:
	rm -f *.o
	rm ReSolve
