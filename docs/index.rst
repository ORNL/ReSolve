*******
ReSolve
*******

ReSolve is an open-source library that provides GPU-resident linear solvers. 
It contains iterative and direct solvers designed to run on NVIDIA and AMD
GPUs, as well as on CPU devices.

ReSolve source code and documentation are available at
`GitHub <https://github.com/ORNL/ReSolve>`_.


Documentation
-------------

To get started, please check our `User Guide <sphinx/user_guide/index.html>`_.
Source code documentation generated in Doxygen is also
`linked <doxygen/html/index.html>`_ to this site.


Contributing
------------

For all contributions to ReSolve please consult 
`Developer Guide <sphinx/developer_guide/index.html>`_ and follow the 
`Coding Style Guidelines <sphinx/coding_guidelines/index.html>`_.

Authors and acknowledgment
--------------------------

Primary authors of this project are:

* Kasia Świrydowicz Kasia.Swirydowicz@amd.com (AMD)
* Slaven Peles peless@ornl.gov (ORNL)

ReSolve project would not be possible without significant contributions
from (in alphabetic order):

* Maksudul Alam (ORNL)
* Ryan Danehy (PNNL)
* Nicholson Koukpaizan (ORNL)
* Jaelyn Litzinger (PNNL)
* Shaked Regev (ORNL)
* Phil Roth (ORNL)
* Cameron Rutherford (PNNL)

Development of this code was supported by the Exascale Computing Project (ECP),
Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations
— the Office of Science and the National Nuclear Security Administration —
responsible for the planning and preparation of a capable exascale ecosystem
— including software, applications, hardware, advanced system engineering, and
early testbed platforms — to support the nation’s exascale computing
imperative.

License
-------

ReSolve is a free software distributed under a BSD-style license. See
the `LICENSE <sphinx/license.html>`__ and `NOTICE <sphinx/notice.html>`__
for more details. All new contributions to ReSolve must be made under the
same licensing terms.

**Please Note:** If you are using ReSolve with any third party libraries linked
in (e.g., KLU), be sure to review the respective license of the package as that
license may have more restrictive terms than the ReSolve license.

Copyright © 2023, UT-Battelle, LLC, and Battelle Memorial Institute.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Resources

   sphinx/user_guide/index
   sphinx/license
   sphinx/notice 


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Developer Resources

   doxygen/index
   sphinx/developer_guide/coding_guidelines
   sphinx/developer_guide/git_guidelines
   sphinx/developer_guide/index
   sphinx/developer_guide/profiling


