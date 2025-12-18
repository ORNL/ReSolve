## Description
 
 <!-- Please describe the issue that is addressed (bug, new feature,
 documentation, enhancement, etc.). Please also include relevant motivation and
 context. List any dependencies that are required for this change.-->
 
 <!-- List issues this PR is closing, e.g. Closes #101 -->
 
 <!-- Mention users/developers who may have interest in this PR, e.g. @supewhiskars --> 
 

 ## Proposed changes
 
 <!-- Describe how your changes here address the issue and why the proposed changes
 should be accepted. -->
 
 ## Checklist
 
 <!-- Put an `x` in the boxes that apply. You can also fill these out after creating
 the PR. If you're unsure about any of them, don't hesitate to ask. We're here
 to help! This is simply a reminder of what we are going to look for before
 merging your code. -->
 

- [ ] All tests pass (`make test` and `make test_install` per testing [instructions](https://github.com/ORNL/ReSolve?tab=readme-ov-file#test-and-deploy)). Code tested on
     - [ ] CPU backend
     - [ ] CUDA backend
     - [ ] HIP backend
- [ ] I have manually run the non-experimental examples and verified that residuals are close to machine precision. (In your build directory run:
`./examples/<your_example>.exe -h` to get instructions how to run examples). Code tested on:
     - [ ] CPU backend
     - [ ] CUDA backend
     - [ ] HIP backend
- [ ] Code compiles cleanly with flags `-Wall -Wpedantic -Wconversion -Wextra`.
- [ ] The new code follows Re::Solve style guidelines.
- [ ] There are unit tests for the new code.
- [ ] The new code is documented.
- [ ] The feature branch is rebased with respect to the target branch.
- [ ] I have updated [CHANGELOG.md](/CHANGELOG.md) to reflect the changes in this PR. If this is a minor PR that is part of a larger fix already included in the file, state so.
 
 ## Further comments
 
 <!-- If this is a relatively large or complex change, kick off the discussion by explaining
 why you chose the solution you did and what alternatives you considered, etc. -->
