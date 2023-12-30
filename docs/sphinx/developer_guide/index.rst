Developer Guide
====================

CMake Build System
-------------------

Our ``cmake`` folder contains some basic CMake modules that help manage resolve:

* ``cmake/FindKLU.cmake``: Our custom find module for KLU that we maintain
* ``cmake/ReSolveConfig.cmake.in``: Our custom config file that is used to generate the ``ReSolveConfig.cmake`` file that is installed with Re::Solve
* ``cmake/ReSolveFindCudaLibraries.cmake``: Our custom find module for CUDA libraries that we maintain to link in subset of cuda needed
* ``cmake/ReSolveFindHipLibraries.cmake``: Our custom find module for HIP/ROCm libraries that we maintain to link in subset of hip needed

Apart from that check out our main ``CMakeLists.txt`` file for our remaining build configuration. 

We also export under the ``ReSolve::`` namespace in our installed CMake configuration for use with ``find_package`` as documented in our main ``README.md``.

Spack Package
---------------

Our current spack package is introduced in https://github.com/spack/spack/pull/40871, and contains support for building Re::Solve with CUDA and HIP/ROCm support.

We also have a custom ``spack`` folder/installation that contains our spack submodule located in ``buildsystem/spack/spack``. This is used to build Re::Solve on CI platforms, as well as support development of the spack package as neccessary.

See the Quik-How-To section below for more information on how to update the spack package and typical workflows for building Re::Solve with spack on CI platforms for testing.


GitHub Actions 
----------------

This is a quick summary of the workflows performed in each GitHub Action. For more information see the ``.github/workflows`` folder where each file is located.

``documentation.yml``
~~~~~~~~~~~~~~~~~~~~~~

``ornl_ascent_mirror.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pushes to ORNL GitLab and triggers CI/CD pipelines there that are posted back to GitHub through commit messages.

``ornl_crusher_mirror.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pushes to ORNL Crusher GitLab...

``pnnl_mirror.yml``
~~~~~~~~~~~~~~~~~~~~

Pushes to PNNL GitLab...

GitLab Pipelines
-----------------

This is a quick summary of the workflows performed in each GitLab Pipeline. For more information see the ``yml`` file for each associated pipeline.

``ornl/crusher.gitlab-ci.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines CI/CD for Crusher at ORNL

``.gitlab-ci.yml``
~~~~~~~~~~~~~~~~~~~~

Located in the root git directory, this defines the CI/CD pipelines for Ascent at ORNL

``pnnl/.gitlab-ci.yml``
~~~~~~~~~~~~~~~~~~~~~~~~

Since single GitLab repo triggers many pipelines as downstream dependents, we need a core config file to kick all of these builds off.

``pnnl/base.gitlab-ci.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Describes core config shared across each job. This could be in ``pnnl/.gitlab-ci.yml`` but we keep it separate for clarity.

``pnnl/deception.gitlab-ci.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deception specific CI.

``pnnl/incline.gitlab-ci.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Incline specific CI.


Writing Documentation
---------------------

Re::Solve uses Sphynx for the documentation. To write and preview the
documentation on your local machine use e.g. ``pip`` tool to install following
Python packages:

.. code:: shell
    
    pip install sphinx docutils sphinx_rtd_theme sphinxcontrib-jquery m2r2

If you prefer using Anaconda utilities, getting these packages is 
slightly different:

.. code:: shell
    
    conda install sphinx docutils sphinx_rtd_theme
    conda install -c conda-forge sphinxcontrib-jquery m2r2


Once you have all the required packages, you can build the HTML docs by

.. code:: shell

  git clone git@github.com:ORNL/ReSolve.git
  sphinx-build -M html ReSolve/docs/ ./build

This will generate HTML documentation and place it in ``build``
subdirectory in your current directory. 


Using Dev Container for Writing Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In case you cannot install Sphynx and other dependencies on your machine,
Re::Solve provides scripts for building development container with all
tools required for Sphynx documentation generation. To create container
for documentation development follow these straightforward steps:

Prerequisites
"""""""""""""

#. install Docker Desktop and launch the app
#. install the "Remote Development" extension in VSCode
#. open your local clone of resolve in VSCode


Build Container
"""""""""""""""

The build info for this container is in `.devcontainer/`. There is a Dockerfile and
json file associated with the configuration.

#. if connected, disconnect from the PNNL VPN
#. launch the container build  

    * `cmd shift p` to open the command pallette in vscode
    * click `> Dev Container: rebuild and reopen container`
    * this will start building the container, taking about 40 minutes
    * click on the pop up with `(show log)` to view the progress

#. Open new terminal within Visual Studio Code and run the renderDocs.sh (note this takes a minute)
#. Open the link that was served to you after step 3

.. note:: Pushing/pulling from git is not supported in a devcontainer,
          and should be done separately.



Devcontainer Configuration
----------------------------

``Dockerfile``
~~~~~~~~~~~~~~

Installs pip and apt dependencies in Python container for doc development.

``devcontainer.json``
~~~~~~~~~~~~~~~~~~~~~~

Configures devcontainer through devcontainer features and sets up extensions.

``renderDocs.sh``
~~~~~~~~~~~~~~~~~~

Small shell script that renders documentation and hosts it for quick development.

Quick How-To guides
-------------------

Re-build Spack tcl modules on CI platforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can rebuild dependencies and spack tcl modules on CI platforms by doing the following using automated CI/CD:

#. If necessary, update spack submodule with latest version of spack:
    #. ``git submodule update --init --recursive``
    #. ``cd buildsystem/spack/spack``
    #. ``git pull checkout develop && git pull``
    #. ``cd ../../..``
    #. ``git add buildsystem/spack/spack``
    #. ``git commit -m "Update spack submodule"``
#. Add a new commit, with a commit message containing build keywords for specific platforms
    * ``[deception-rebuild]`` for Deception
    * ``[incline-rebuild]`` for Incline
    * ``[crusher-rebuild]`` for Crusher
    * ``[ascent-rebuild]`` for Ascent
#. Push to GitHub within an existing PR

Once you do this, each GitLab site that has a build triggered will do the following:

#. With the updated submodule, rebuild resolve alongside new dependencies
#. Push the new spack tcl modules back to the GitHub repo with a commit message containing test keywords for the specific platform
#. Run tests as commits are added for each platform as modules are re-built

Alternatively, you can log-on to the cluster of choice, and also build/iterate by hand with this workflow:

#. Log-on to the cluster of choice, update submodules
#. Load spack environment with ``. buildsystem/load-spack.sh``
#. Concretize and test the config you want to install is correct with ``spack concretize -f``, making changes as necessary
#. Install configuration and tcl modules with ``./buildsystem/configure-modules.sh``

Each cluster also supports submitting this job to the job scheduler by doing the following:

* ``./buildsystem/spack/<cluster>/install.sh`` to run the workflow as-is on the current node
* ``sbatch ./buildsystem/spack/<cluster>/install.sh`` to submit the workflow to the job scheduler

Update Re::Solve spack package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to test any modifications to the spack package, it is suggested that you use the above automated workflows to make testing as seamless as possible.

When trying to upstream changes that you want to make to Re::Solver's spack package, you can do the following:

#. Fork the spack repo to your own GitHub account / another location
#. Use this fork as the submodule in ``buildsystem/spack/spack``
#. Make changes to the spack package as necessary after loading spack using ``spack edit resolve``
#. Commit changes to your forked spack repo
#. Update the submodule in Re::Solve with the new commit hash, push updated submodule to GitHub and test on CI platforms as described above
#. Once you are satisfied with the changes, submit a PR to the main spack repo with the changes

Typically this is done as a part of the release process, so also make sure that you follow the release checklist where appropriate.

Note that spack enforces it's own styling, so consider leveraging spack-bot in order to help out. Comment ``@spackbot help`` to get a list of commands, such as ``@spackbot fix style`` to have it try and automatically style your PR for you!

Refresh GitHub/GitLab Secrets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several secrets throughout CI/CD:

* GitHub commit status tokens for posting back to GitHub from GitLab at:
    * PNNL (Deception, Incline)
    * ORNL (Crusher, Ascent)
* GitHub push tokens for update spack build tcl modules from GitLab at:
    * PNNL (Deception, Incline)
    * ORNL (Crusher, Ascent)
* GitLab tokens to allow push mirror from GitHub to GitLab at:
    * PNNL (Deception, Incline)
    * ORNL (Crusher, Ascent)

These 6 tokens in total are all generated with different permission scops, and across both GitLab and GitHub. They are stored as separate secrets across all the repositories.

Ensure not to re-use tokens for multiple purposes, and if a token is ever exposed over plaintext, it should be re-generated ASAP.
