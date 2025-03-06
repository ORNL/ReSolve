Profiling
=========



##########
HPCToolkit
##########

`HPCToolkit <http://hpctoolkit.org>`_ can profile any code built with debug
symbols, it does not require special code instrumenting. In CMake, one simply
needs to select ``RelWithDebInfo`` as the build type.

Consult HPCToolkit `documentation <http://hpctoolkit.org/software-instructions.html>`_
to find out how to obtain and install all the required profiling software.
Hopefully, your system administrator will install HPCToolkit for you and you
would only need to load appropriate modules.

.. note:: On Frontier supercomputer all you need to get are ``ums``,
          ``ums023`` and ``hpctoolkit`` modules.

Once you built your code with debug symbols and got the working HPCToolkit
installation, you run profiling with

.. code:: shell

  hpcrun -t -e CPUTIME -e gpu=amd -o out.m my_executable.exe

`hpcrun <http://hpctoolkit.org/man/hpcrun.html>`_ is a profiling tool that
collects call path profiles of program executions using statistical sampling
of hardware counters, software counters, or timers. In the example above:

  * Flag ``-t`` tells profiler to generate a call path trace in addition to a
    call path profile.
  * Flag ``-e`` selects events or sampling periods for profiling.

    * ``-e CPUTIME`` will tell profiler to use Linux system timer to set the
      sampling rate.
    * ``-e gpu=amd`` will instruct profiler to collect comprehensive
      operation-level measurements for HIP programs on AMD GPU.

  * Flag ``-o`` specifies the directory where the output data will be stored.
  * In this example, ``my_executable.exe`` is the binary being profiled.

It is often helpful to map profiling data to specific CPU and GPU code. This
can be done by invoking `hpcstruct <http://hpctoolkit.org/man/hpcstruct.html>`_
command as

.. code:: shell

  hpcstruct out.m

where ``out.m`` is the directory with profiling measurements collected by
``hpcrun``. This step is optional.

`hpcprof <http://hpctoolkit.org/man/hpcprof.html>`_ analyzes profile
performance measurements and attributes them to static source code structure.
One can simply call

.. code:: shell

  hpcprof -o out.d out.m

This will generate profiling information using measurements in directory
``out.m`` and store it in directory ``out.d``. The data in ``out.d`` can be
viewed using ``hpcviewer`` tool.

It is recommended to download and install graphical version of
`HPCToolkit viewer <http://hpctoolkit.org/download.html>`_ on your local
machine and analyze profiling data there. The graphical user interface provides
many productivity features but often runs slowly over SSH connection. 

When running the profiler on a machine with a scheduler, it is best to use
a script. When using SLURM, for example, a script to run the profiler would
look something like this:

.. code:: shell

  #!/bin/bash

  #SBATCH -A MyAllocation
  #SBATCH -J resolve_profile
  #SBATCH -o %x-%j.out
  #SBATCH -t 00:30:00
  #SBATCH -N 1
  
  EXE=resolve_executable.exe
  OUT=hpctoolkit_resolve_profile

  # Profile ReSolve code on a single GPU  
  echo "`date` Starting run"
  srun -N 1 -n 1 -c 1 -G 1 \
    hpcrun -t -e CPUTIME -e gpu=amd -o ${OUT}.m ${EXE}
  echo "`date` Finished run"
  
  # Use 56 cores to process profiling measurements
  srun -N 1 -n 1 -c 56 hpcstruct ${OUT}.m
  srun -N 1 -n 1 -c 56 hpcprof -o ${OUT}.d ${OUT}.m

Any arguments that the executable takes can be simply added as when the
executable is called locally. Similar script could be written for LSF or MOAB
scheduler.

###############
AMD ROCProfiler
###############

To profile GPU kernels in the HIP backend, we recommend using AMD tools. Since ``ROCm 6.2``, 
the recommended approach is `ROCprofiler-SDK <https://github.com/rocm/rocprofiler-sdk>`_. 
ROCprofiler-SDK is part of the ROCm library, so no additional software needs to be installed 
once you obtain ROCm. 

ROCTX annotations are useful to place kernel execution in the context of CPU code execution. 
These can be added as follow:

.. code:: c++

  // some include files ...

  #include <rocprofiler-sdk-roctx/roctx.h>

  // some code ...

  roctxRangePush("My Event");

  // my event code ...

  roctxRangePop();
  roctxMarkA("My Event");

The string label is an optional argument to the annotation code.

At this time, Re::Solve implements the macros ``RESOLVE_RANGE_PUSH`` and ``RESOLVE_RANGE_POP`` 
to select ``NVTX`` or ``ROCTX`` with the appropriate ``nvtxRangePush`` or ``roctxRangePush``, 
respectively, depending on the active backend. Make sure your Re::Solve library was built with 
profiling support, i.e. the build was configured with CMake boolean flag ``RESOLVE_USE_PROFILING`` 
set to ``On``. The appropriate profiling library will be linked depending on the backend selected.

Once your instrumented code is built, it can be profiled as follows:

.. code:: shell

  rocprofv3 --stats --hip-trace --roctx-trace -o out.csv ./my_executable.exe

In this example

  * Flag ``-o`` specifies the output file in comma separated values format,
    in this case ``out.csv``.
  * File ``my_executable.exe`` is the binary being profiled.
  * Flag ``--stats`` enables kernel execution stats.
  * Flag ``--hip-trace`` includes HIP API timelines in profiling data.
  * Flag ``--roctx-trace`` enables rocTX application code annotation trace.

The profiler will create several files with name ``out`` but with different
extensions. To visualize output, one can upload the ``out.json`` file to
`Perfetto <https://ui.perfetto.dev/>`_.

When running ROCProfiler on a machine with a scheduler, it is a good idea
to write a profiling script. Here is an example for a SLURM scheduler:

.. code:: shell

  #!/bin/bash                                                                                                                                

  #SBATCH -A CSC359                                                                                                                          
  #SBATCH -J resolve_test                                                                                                                    
  #SBATCH -o %x-%j.out                                                                                                                       
  #SBATCH -t 00:30:00                                                                                                                        
  #SBATCH -N 1                                                                                                                               
  
  EXE=build/examples/klu_rocsolverrf_fgmres.exe
  OUT=rocprof-resolve25k
  ARGS=""
  
  echo "`date` Starting run"
  srun -N 1 -n 1 -c 1 -G 1 \
    rocprofv3 --stats --hip-trace --roctx-trace -o ${OUT}.csv \
    ${EXE} ${ARGS}
  echo "`date` Finished run"

#####################
NVIDIA Nsight Systems
#####################
Similar to ROCTX annotations, `NVTX <https://github.com/NVIDIA/NVTX>`_ annotations are useful 
to place kernel execution in the context of CPU code execution when using the CUDA backend.

You can annotate events you want to trace in your code execution as follows:

.. code:: c++

  // some include files ...

  #include <nvToolsExt.h>

  // some code ...

  nvtxRangePush("My Event");

  // my event code ...

  nvtxRangePop();
  nvtxMarkA("My Event");

Multiple tools support NVTX annotations. We recommend 
`Nsight Systems <https://developer.nvidia.com/nsight-systems>`_ for a high level profile. 
It can be used as follows:

.. code:: shell

  nsys profile --stats=true --trace=cuda,nvtx ./my_executable.exe

This will generate a text output with high level execution statistics (time spent in kernels, 
NVTX ranges, memory)  as well as ``*.nsys-rep`` and ``*.sqlite`` files. The ``*nsys-rep`` 
can be opened in the Nsight System graphical user interface to visualize the timeline. 
You will need to download the file to a local machine if you are profiling on the remote 
system that does not support graphical user interfaces.
