
Code Style
==========

Error handling
--------------

Return values of member functions should be of type ``int`` and used for
error handling. Functions return 0 if no error is encountered, return
positive value for warnings and recoverable error, and negative value
for irrecoverable errors.

Output
------

If an output is needed (for example, a warning needs to be displayed),
use ``std::cout`` and not ``printf`` as shown below. There should be a
space before and after each ``<<``.

.. code:: cpp

   std::cout << "index out of bounds. Row " << i << " starts at: " << start << " and ends at " << end << std::endl;

General naming conventions
--------------------------

Do not include words like "function", "variable", "const" or "class" in names. It is clear what they are based on the naming conventions.

Member variable naming
------------------------------

Member variable names should use C-style name format and end with
trailing underscore ``_``.

.. code:: cpp

   double member_variable_; // Yes
   double another_member;   // No, there is no trailing underscore to distinguish it from nonmember variables
   double memberVariable_;  // No, using lowercase camel instead of C-style name format

Function names
--------------

Use lowercase camel format (camelCase), with no underscore, for function names.

.. code:: cpp

   int myFunction(double x); // Yes
   int another_function();   // No, using C-style name format
   int YetAnotherFunction(); // No, using uppercase camel name format

Class names
-----------

Class names should follow uppercase camel format (CamelCase), with no underscore. 
For instance, `Vector` and `CsrMatrix` are valid class names, while `cartesianPoint` is not.

Enums (enumerated types)
------------------------------

Always define ``enum``\ s inside ``ReSolve`` namespace. Type names
should be capitalized and the constant names should be uppercase with
underscores (but there is no underscore at the end!).

.. code:: cpp

     enum ExampleEnum { CONST_ONE = 0,
                        CONST_TWO = 8, 
                        YET_ANOTHER_CONST = 17 };

Constants
---------

If a constant is used in more than one file, define it in ``Common.h``.
Constants names should be capitalized and words separated by underscores.

.. code:: cpp

      constexpr double Pi = 3.1415; // No, it should be all caps
      constexpr double SQRT_TWO = 1.4142 // Yes
      constexpr double SQRTTWO_ = 1.4142 // No, there is a trailing underscore but not between words
      constexpr double EXP = 2.7183 // Yes  

Exceptions to naming conventions
--------------------------------

The following are exceptions to the naming conventions:
Always capitalize `ReSolve` in this manner. There may be additional words
where capitalization is important and it is preserved similarly.

Capitalize variables refering to a matrix `A`
such as `A`, `At`, `A_csc`, and `A_csr`.
This is due to the equation $Ax=b$ and the Householder convention where
uppercase letters represent matrices and lowercase letter represent vectors.

Pointers and references
------------------------------

The pointer ``*`` or reference ``&`` belong to the type and there should
be no space between them and the type name.
Declare each variable on a different line.

.. code:: cpp

   double* x;     // Yes
   int& n;        // Yes
   double *x, *y; // No, the pointer symbol is a part of `double*` type
   int & n;       // No, the reference symbol is a part of `int&` type

Indentation
-----------

When writing a function header over multiple lines, 
indenting of subsequent variables should mimic the first variable.
Use only spaces for indentation, not tabs. Indent size is 2 spaces.

When defining a class, the code blocks after ``private``, ``public`` and
``protected`` should be indented. There should be an empty line before
each definition (except the first one). See example below.

.. code:: cpp

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

Braces
------

Namespaces, classes and functions: use new line afterwards, i.e.,

.. code:: cpp

   namespace someNamespace
   {
     //some code 
   }

For short functions (i.e., empty constructor), do not inline braces.

.. code:: cpp

   ClassA::ClassA()
   {
   }

Have opening brace at the next line after the ``for``, ``if``, or ``while``
statement (Allman style). See, e.g, the example below:

.. code:: cpp

   if (cond == true)
   {
     // some code
   }
   else
   {
     // some other code
   }

Have a space between keywords ``for``, ``while`` and ``if`` and the
parenthesis as shown here:

.. code:: cpp

   for (int i = 0; i < n; ++i)
   {
     // some code
   } 

Do not use one-line ``for``\ s. Always use braces.
One-line ``if``\ s are acceptable (though not mandatory) if the command directly follows from the conditional.
I.e. the command can only be executed if the conditional is true.
An ``if`` statement where the conditional and the command are not fundamentally connected should use braces.
The following are correct uses:

.. code:: cpp

   if (owns_cpu_data_ && h_data_) delete [] h_data_; // Ok. The intent is obvious.
   if (x != 0) z = y / x;  // Ok. Operation is always invalid, regardless of code structure.
   if (y % 2 == 1) {       // Must be written this way, the logic is internal to the code.
      x++;
   }




Use of spaces and newlines
------------------------------

There should be spaces between arithmetic operators.

.. code:: cpp

   x = c * (a + b);  //Yes
   x = c*(a+b).      // No, the clarity is better with spaces around binary operators.

When defining member functions, use one empty line between the
functions.

.. code:: cpp

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

Leave one empty line between all the includes and the first line of the
actual code.

.. code:: cpp

   #include <iostream>

   int main()
   {
     std::cout 
   }

Also, leave one empty line between ``system`` includes and ``resolve``
includes, i.e.,

.. code:: cpp

   #include <cstring>

   #include <resolve/matrix/Coo.hpp>

   int main()
   {
     //some code
     return 0;
   }

The ``system`` includes should always be listed first.

Using namespaces
----------------

All classes should be in namespace ``ReSolve``. If needed, define
additional namespaces inside ``ReSolve``.

.. code:: cpp
   
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

Writing comments
----------------

Use ``//`` for comments in the code. Do not use ``/* */``.

Put Doxygen comments for functions in their source file. 
Put Doxygen comments for classes in their header file.

Use this format for comments:

.. code:: cpp
   
   /**
    *
    */

Write a brief description of the function using ``@brief``.
Add a longer description below, if needed.

Define conditions for 

For each parameter, use ``@param`` to describe it.
Explain which parameters are in, out or in,out using 
``@param[in]``, ``@param[out]`` and ``@param[in,out]``.

Explain the return value using ``@return``.

Overall, the comment should look like this:

.. code:: cpp

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
       * @return 0 if no error, positive value for warnings and recoverable error, 
       * negative value for irrecoverable errors
      */
      void logAdd(const real_type x, real_type& y, real_type* z)
      {
        y = log(x);
        *z += y;
      }

Do not leave commented code used for debugging (or for other purposes).
Remove it before committing the code.

At the developer's and reviewer's discretion, trivial functions can have one
line documentation with ``///`` or none at all. Examples of trivial functions
are getters and setters, or functions that are self-explanatory.
For example a getter function can be documented as follows:

.. code:: cpp

   /// This function returns the x coordinate - this line can be omitted
   double getX() const 
   { 
     return x_; 
   } 
