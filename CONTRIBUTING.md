# Developer Guidelines

## Code Style

### Error handling

Return values of member functions should be of type `int` and used for
error handling. Functions return 0 if no error is encountered, return
positive value for warnings and recoverable error, and negative value
for irrecoverable errors.

### Output

If an output is needed (for example, a warning needs to be displayed),
use `std::cout` and not `printf` as shown below. There should be a
space before and after each `<<`.

```cpp
std::cout << "index out of bounds. Row " << i << " starts at: " << start << " and ends at " << end << std::endl;
```

### General naming conventions

Do not include words like "function", "variable", "const" or "class" in names. It is clear what they are based on the naming conventions.

### Member variable naming

Member variable names should use C-style name format and end with
trailing underscore `_`.

```cpp
double member_variable_; // Yes
double another_member;   // No, there is no trailing underscore to distinguish it from nonmember variables
double memberVariable_;  // No, using lowercase camel instead of C-style name format
```

### Function names

Use lowercase camel format (camelCase), with no underscore, for function names.

```cpp
int myFunction(double x); // Yes
int another_function();   // No, using C-style name format
int YetAnotherFunction(); // No, using uppercase camel name format
```

### Class names

Class names should follow uppercase camel format (CamelCase), with no underscore. 
For instance, `Vector` and `CsrMatrix` are valid class names, while `cartesianPoint` is not.

### Enums (enumerated types)

Always define `enum`s inside `ReSolve` namespace. Type names
should be capitalized and the constant names should be uppercase with
underscores (but there is no underscore at the end!).

```cpp
enum ExampleEnum { CONST_ONE = 0,
                   CONST_TWO = 8, 
                   YET_ANOTHER_CONST = 17 };
```

### Constants

If a constant is used in more than one file, define it in `Common.h`.
Constants names should be capitalized and words separated by underscores.

```cpp
constexpr double Pi = 3.1415; // No, it should be all caps
constexpr double SQRT_TWO = 1.4142 // Yes
constexpr double SQRTTWO_ = 1.4142 // No, there is a trailing underscore but not between words
constexpr double EXP = 2.7183 // Yes  
```

### Exceptions to naming conventions

The following are exceptions to the naming conventions:
Always capitalize `ReSolve` in this manner. There may be additional words
where capitalization is important and it is preserved similarly.

Capitalize variables refering to a matrix `A`
such as `A`, `At`, `A_csc`, and `A_csr`.
This is due to the equation $Ax=b$ and the Householder convention where
uppercase letters represent matrices and lowercase letter represent vectors.

### Pointers and references

The pointer `*` or reference `&` belong to the type and there should
be no space between them and the type name.
Declare each variable on a different line.

```cpp
double* x;     // Yes
int& n;        // Yes
double *x, *y; // No, the pointer symbol is a part of `double*` type
int & n;       // No, the reference symbol is a part of `int&` type
```

### Indentation

When writing a function header over multiple lines, 
indenting of subsequent variables should mimic the first variable.
Use only spaces for indentation, not tabs. Indent size is 2 spaces.

When defining a class, the code blocks after `private`, `public` and
`protected` should be indented. There should be an empty line before
each definition (except the first one). See example below.

```cpp
class SomeClass
{
  public:
    SomeClass();
    ~SomeClass();

  private:
    int some_variable_;

  protected:
    void someFunction();
};
```

### Braces

Namespaces, classes and functions: use new line afterwards, i.e.,

```cpp
namespace someNamespace
{
  //some code 
}
```

For short functions (i.e., empty constructor), do not inline braces.

```cpp
ClassA::ClassA()
{
}
```

Have opening brace at the same line as the `for`, `if`, or `while`
statement. Leave a space between the statement and the brace. When using
`else`, follow the example below.

```cpp
if (cond == true) {
  // some code
} else {
  // some other code
}
```

Have a space between keywords `for`, `while` and `if` and the
parenthesis as shown here:

```cpp
for (int i = 0; i < n; ++i) {
  // some code
} 
```

Do not use one-line `for`s. Always use braces.
One-line `if`s are acceptable (though not mandatory) if the command directly follows from the conditional.
I.e. the command can only be executed if the conditional is true.
An `if` statement where the conditional and the command are not fundamentally connected should use braces.
The following are correct uses:

```cpp
if (owns_cpu_data_ && h_data_) delete [] h_data_; // Ok. A method should only delete something if it is allocated and the class owns it.
if (x != 0) z = y / x; // Ok. Division by 0 is always invalid, regardless of code structure.
if (y % 2 == 1) { //Must be written this way, because x can be incremented regardless of y's parity. The logic is internal to our code.
   x++;
}
```

### Use of spaces and newlines

There should be spaces between arithmetic operators.

```cpp
x = c * (a + b);  //Yes
x = c*(a+b).      // No, the clarity is better if there are spaces between binary operators and operands.
```

When defining member functions, use one empty line between the
functions.

```cpp
struct MyStruct
{
  int memberFunction()
  {
    // some code
  }

  int anotherMemberFunction()
  {
    // some other code
  }
};
```

Leave one empty line between all the includes and the first line of the
actual code.

```cpp
#include <iostream>

int main()
{
  std::cout 
}
```

Also, leave one empty line between `system` includes and `resolve`
includes, i.e.,

```cpp
#include <cstring>

#include <resolve/matrix/Coo.hpp>

int main()
{
  //some code
  return 0;
}
```

The `system` includes should always be listed first.

### Using namespaces

All classes should be in namespace `ReSolve`. If needed, define
additional namespaces inside `ReSolve`.

```cpp
namespace ReSolve 
{
   class Solver // Yes, class defined inside ReSolve namespace 
   { 
      // some code; 
   };

   namespace LinearAlgebra 
   { 
      class Vector // Yes, class defined inside ReSolve namespace 
      { 
         // vector code 
      }; 
   } 
}

class Matrix // No, class is outside ReSolve namespace 
{ 
   // matrix code
};
```

### Writing comments

Use `//` for comments in the code. Do not use `/* */`.

Put Doxygen comments for functions in their source file. 
Put Doxygen comments for classes in their header file.

Use this format for comments:

```cpp
/**
 *
 */
```

Write a brief description of the function using `@brief`.
Add a longer description below, if needed.

Define conditions for 

For each parameter, use `@param` to describe it.
Explain which parameters are in, out or in,out using 
`@param[in]`, `@param[out]` and `@param[in,out]`.

Explain the return value using `@return`.

Overall, the comment should look like this:

```cpp
/**
 * @brief This computes the logarithm of x
 *
 * The function stores the logarithm of x in y and adds it to z.
 * 
 * @pre x > 0
 * @post y = log(x)
 * @post z = z + y
 * @invariant x
 * 
 * @param[in] x The input parameter
 * @param[out] y The output parameter
 * @param[in,out] z The input/output parameter
 *
 * @return 0 if no error, positive value for warnings and recoverable error, negative value for irrecoverable errors
*/
void logAdd(const real_type x, real_type& y, real_type* z)
{
  y = log(x);
  *z += y;
}
```

Do not leave commented code used for debugging (or for other purposes).
Remove it before committing the code.

At the developer's and reviewer's discretion, trivial functions can have one
line documentation with "//" or none at all. Examples of trivial functions
are getters and setters, or functions that are self-explanatory.
For example a getter function can be documented as follows:

```cpp
// This function returns the x coordinate - this line can be omitted
double getX() const 
{ 
  return x_; 
} 
```

# Git Guidelines

## Branching and workflow

Define the scope of the changes you are working on. If you notice unrelated issues, create a separate branch or new issue to address them.

When you don't expect changes on your branch to clash with other changes, branch off the main development branch (usually `develop`).

Otherwise, branch off the feature branch that you are working on.

Name the branch:

`<your_name>/<short_description>`

e.g.

`jane_doe/typo_fix`

When working on a large feature, create a feature development branch. Then break it down into smaller tasks and create a branch for each task (branching off the main feature branch). Periodically rebase the feature branch with respect to `develop` to keep it up to date.

## Opening an Issue

When opening an issue, check that it is not a duplicate. Provide a clear and concise description, using the issue template.

## Pull Requests (PR)s

When opening a PR, make sure to select the correct base branch. If there are merge conflicts, use `git rebase` to resolve them before requesting a review.

Detail changes made in your PR in the description.

1. What new features were added?
2. What bugs were fixed?
3. What tests were added?

When reviewing a PR consider the following:

1. Does the PR address the issue? (if applicable)
2. Do existing and new tests pass? (Run the code on different machines)
3. Is the code clean, readable, and does it have proper comments?
4. Are the changes consistent with the coding guidelines?

### Minor concerns:

Add a comment to the PR with the minor concern and request the author to address it, but approve the merge.

### Major concern options:

1. Suggest a change with the Github suggest feature within the PR for the author to commit before merging.
2. Request the author to make the change and wait to approve the merge.
3. Branch off the PR, make the change, and submit a new PR with the change. Make the assignee the author of the current PR and request the author to merge the new PR.

### Merging

All PRs must pass the checks and a review (by someone other than the PR author) before merging. Once the PR is approved and the checks pass, the author can merge the PR.

If the author does not have permission to merge, the reviewer can merge the PR. Use squash merge when merging a feature to a main development branch. Edit the commit message to describe relevant details. When a PR is merged, delete the branch.RetryClaude does not have the ability to run the code it generates yet.
