*******
Re::Solve
*******

Re::Solve is an open-source library that provides GPU-resident linear solvers. 
It contains iterative and direct solvers designed to run on NVIDIA and AMD
GPUs, as well as on CPU devices.

Re::Solve source code and documentation are available at
`GitHub <https://github.com/ORNL/ReSolve>`_.


Documentation
-------------

To get started, please check our `User Guide <sphinx/user_guide/index.html>`_.
Source code documentation generated in Doxygen is also
`linked <doxygen/html/index.html>`_ to this site.


Support
-------
For technical questions or to report a bug please submit a
`GitHub issue <https://github.com/ORNL/ReSolve/issues>`_ or post a question on
`user mailing list <mailto:resolve-users@elist.ornl.gov>`_.
For non-technical issues please contact
`Re::Solve developers <mailto:resolve-devel@elist.ornl.gov>`_.



Contributing
------------

For all contributions to Re::Solve please consult 
`Developer Guide <sphinx/developer_guide/index.html>`_ and follow the 
`Coding Style Guidelines <sphinx/coding_guidelines/index.html>`_.

Authors and acknowledgment
--------------------------

The primary authors of this project are:

* Kasia Świrydowicz Kasia.Swirydowicz@amd.com (AMD)
* Slaven Peles peless@ornl.gov (ORNL)
* Shaked Regev regevs@ornl.gov (ORNL)

Re::Solve project would not be possible without significant contributions
from (in alphabetic order):

* Maksudul Alam (ORNL)
* Kaleb Brunhoeber (ORNL)
* Ryan Danehy (PNNL)
* Adham Ibrahim (ORNL)
* Nicholson Koukpaizan (ORNL)
* Jaelyn Litzinger (PNNL)
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

Re::Solve is a free software distributed under a BSD-style license. See
the `LICENSE <sphinx/license.html>`__ and `NOTICE <sphinx/notice.html>`__
for more details. All new contributions to Re::Solve must be made under the
same licensing terms.

**Please Note:** If you are using Re::Solve with any third party libraries linked
in (e.g., SuiteSparse), be sure to review the respective license of the package as that
license may have more restrictive terms than the Re::Solve license.

Copyright © 2023, UT-Battelle, LLC, and Battelle Memorial Institute.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Resources

   sphinx/user_guide/index
   sphinx/license
   sphinx/notice 


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Resources

   sphinx/developer_guide/index
   doxygen/index
   sphinx/developer_guide/profiling
