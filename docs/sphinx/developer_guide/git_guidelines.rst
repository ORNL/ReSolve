
Branching and workflow
------------
Workflows

Git Branching
-------------

Define the scope of the changes you are working on. If you notice unrelated
issues, create a separate branch or new issue to address them.

When you don't expect changes on your branch to clash with other changes, 
branch off the main development branch (usually `develop`). 
Otherwise, branch off the feature branch that you are working on.

Name the branch: 
``<your_name>/<short_description>`` 
e.g. 
``jane_doe/typo_fix``

When working on a large feature, create a feature development branch.
Then break it down into smaller tasks and create a branch for each task
(branching off the main feature branch). Periodically rebase the feature branch
with respect to `develop` to keep it up to date.


Creating a GitHub Issue
------------

When creating an issue, check that it is not a duplicate.
Provide a clear and concise description, using the issue template.

GitHub Pull Requests 
------------

When creating a pull request (PR), make sure to select the correct base branch.
If there are merge conflicts, use ``git rebase`` to resolve them before requesting a review.

Detail changes made in your PR in the description.
^^^^^^^^^^^^^^^^^^
1. What new features were added?
2. What bugs were fixed?
3. What tests were added?

When reviewing a PR consider the following:
^^^^^^^^^^^^^^^^^^
1. Does the PR address the issue?
2. Do existing and new tests pass on all supported platforms?
3. Is the code clean, readable, and properly documented?
4. Does the code follows Re::Solve style guidelines?
5. Does the code compiles cleanly with flags `-Wall -Wpedantic -Wconversion -Wextra`.
6. Are there unit tests for the new code.
7. Is the feature branch is rebased with respect to the target branch.


Minor concerns:
^^^^^^^^^^^^^^^^^^
Add a comment to the PR with the minor concern and request the author to address it, but approve the merge.


Major concern options:
^^^^^^^^^^^^^^^^^^
1. Suggest a change with the Github suggest feature within the PR for the author to commit before merging.
2. Request the author to make the change and wait to approve the merge.
3. Branch off the PR, make the change, and submit a new PR with the change. Make the assignee the author of the current PR and request the author to merge the new PR.

Merging
^^^^^^^^^^^^^^^^^^
All PRs must pass the checks and a review 
(by someone other than the PR author) before merging.
Once the PR is approved and the checks pass, the author can merge the PR. 
If the author does not have permission to merge, the reviewer can merge the PR.
Use squash merge when merging a feature to a main development branch.
Edit the commit message to describe relevant details.
When a PR is merged, delete the branch.
