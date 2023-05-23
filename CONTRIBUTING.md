# Developer Guidelines

## Code Style


### Error handling
Return values of member functions should be of type `int` and used for error handling. Functions return 0 if no error is encounter, return positive value for warnings and recoverable error, and negative value for irrecoverable errors.

### Member variable naming

Member variable names should use C-style name format and end with trailing underscore `_`.
```c++
double member_variable_; // Yes
double another_member;   // No, there is no trailing underscore to distinguish it from nonmember variables
double memberVariable_;  // No, using lowercase camel instead of C-style name format
```

### Function names

Use lowercase camel format for function names.
```c++
int myFunction(double x); // Yes
int another_function();   // No, using C-style name format
int YetAnotherFunction(); // No, using uppercase camel name format
```

### Pointers and references

The pointer `*` or reference `&` belong to the type and there should be no space between them and the type name.
```c++
double* x;     // Yes
int& n;        // Yes
double *x, *y; // No, the pointer symbol is a part of `double*` type
int & n;       // No, the reference symbol is a part of `int&` type
```

### Indentation
Use only spaces for indentation, not tabs. Indent size is 2 spaces.

### Braces
Namespaces, classes and functions: use new line afterwards, i.e.,  
```c++
namespace someNamespace
{
  //some code 
}
```
For short functions (i.e., empty constructor), do not inline braces.
```c++
classA::classA()
{
}
```
Have opening brace at the same line as the  `for`, `if`, or `while` statement. Leave a space between the statement and the brace. When using `else`, follow the example below. 
```c++
if (cond == true) {
  // some code
} else {
  // some other code
}
 ```
Do not use one-line `if`s and `for`s. Always use braces.

### Use of spaces and newlines
There should be spaces between arithmetic operators. 
```c++
x = c * (a + b);  //Yes
x = c*(a+b).      // No, the clarity is better if there are spaces between binary operators and operands.
```
When defining member functions, use one empty line between the functions.
```c++
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
Leave one empty line between all the includes and the first line of the actual code. 
```c++
#include <iostream>

int main()
{
  std::cout 
}
```


### Using namespaces
All classes should be in namespace `ReSolve`. If needed, define additional namespaces inside `ReSolve`.
```c++
namespace ReSolve
{
  class Solver  // Yes, class defined inside ReSolve namespace
  {
    // some code; 
  };

  namespace LinearAlgebra
  {
    class Vector  // Yes, class defined inside ReSolve namespace
    {
      // vector code
    };
  }
}

class Matrix   // No, class is outside ReSolve namespace
{
  // matrix code
};


