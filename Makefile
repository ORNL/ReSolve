CC := g++
NVCC := nvcc
CUDA_LIBS := -lcusparse -lcublas -lcusolver
LIBS := -lklu
SRCCPU := resolveMatrixHandler.cpp  resolveVectorHandler.cpp resolveMatrixIO.cpp resolveLinSolver.cpp resolveLinSolverDirectKLU.cpp main.cpp
OBJSCPU := $(foreach f,$(SRCCPU),$(f:%.cpp=%.o))

SRCCUDA := resolveMatrix.cpp resolveVector.cpp resolveLinAlgWorkspace.cpp resolveLinSolverDirectCuSolverRf.cpp

OBJSCUDA := $(foreach f,$(SRCCUDA),$(f:%.cpp=%.o))

all: ${OBJSCPU} ${OBJSCUDA} ReSolve

%.o: %.cpp
	${CC} ${CUDALIBS} -o $@ -c $<
ReSolve: ${OBJCPU} ${OBJGPU}
	${NVCC} -o $@ ${OBJSCPU} ${OBJSCUDA} ${LIBS} ${CUDA_LIBS}

clean:
	rm -f *.o
	rm ReSolve
