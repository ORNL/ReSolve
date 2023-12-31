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

Similar script could be written for LSF or MOAB scheduler.

###########
ROCProfiler
###########

Unlike HPCToolkit, `ROCProfiler <https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof.html>`_
requires code to be instrumented.

