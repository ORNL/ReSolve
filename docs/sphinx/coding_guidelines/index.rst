Code Style Guidelines
======================

Code Style
----------

Error handling
~~~~~~~~~~~~~~

Return values of member functions should be of type ``int`` and used for
error handling. Functions return 0 if no error is encounter, return
positive value for warnings and recoverable error, and negative value
for irrecoverable errors.

Output
~~~~~~

If an output is needed (for example, a warning needs to be displayed),
use ``std::cout`` and not ``printf`` as shown below. There should be a
space before and after each ``<<``.

.. code:: cpp

   std::cout << "index out of bounds. Row " << i << " starts at: " << start << " and ends at " << end << std::endl;

Member variable naming
~~~~~~~~~~~~~~~~~~~~~~

Member variable names should use C-style name format and end with
trailing underscore ``_``.

.. code:: cpp

   double member_variable_; // Yes
   double another_member;   // No, there is no trailing underscore to distinguish it from nonmember variables
   double memberVariable_;  // No, using lowercase camel instead of C-style name format

Function names
~~~~~~~~~~~~~~

Use lowercase camel format for function names.

.. code:: cpp

   int myFunction(double x); // Yes
   int another_function();   // No, using C-style name format
   int YetAnotherFunction(); // No, using uppercase camel name format

Class names
~~~~~~~~~~~

Class names should start with a capital letter. For instance, ``Vector``
and ``Matrix`` are valid class names, while ``point`` is not.

Enums (enumerated types)
~~~~~~~~~~~~~~~~~~~~~~~~

Always define ``enum``\ s inside ``ReSolve`` namespace. Type names
should be capitalized and the constant names should be uppercase with
underscores (but there is no underscore at the end!).

.. code:: cpp

     enum ExampleEnum { CONST_ONE = 0,
                        CONST_TWO = 8, 
                        YET_ANOTHER_CONST = 17 };

Constants
~~~~~~~~~

If a constant is used in more than one file, define it in ``Common.h``.
Constants names should be capitalized.

.. code:: cpp

      constexpr double Pi = 3.1415; // No, it should be all caps
      constexpr double SQRT_TWO = 1.4142 // No, there is an underscore
      constexpr double SQRTTWO_ = 1.4142 // No, there is an underscore
      constexpr double EXP = 2.7183 // Yes   

Pointers and references
~~~~~~~~~~~~~~~~~~~~~~~

The pointer ``*`` or reference ``&`` belong to the type and there should
be no space between them and the type name.

.. code:: cpp

   double* x;     // Yes
   int& n;        // Yes
   double *x, *y; // No, the pointer symbol is a part of `double*` type
   int & n;       // No, the reference symbol is a part of `int&` type

Indentation
~~~~~~~~~~~

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
~~~~~~

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

Have opening brace at the same line as the ``for``, ``if``, or ``while``
statement. Leave a space between the statement and the brace. When using
``else``, follow the example below.

.. code:: cpp

   if (cond == true) {
     // some code
   } else {
     // some other code
   }

Have a space between keywords ``for``, ``while`` and ``if`` and the
parenthesis as shown here:

.. code:: cpp

   for (int i = 0; i < n; ++i) {
     // some code
   } 

Do not use one-line ``if``\ s and ``for``\ s. Always use braces.

Use of spaces and newlines
~~~~~~~~~~~~~~~~~~~~~~~~~~

There should be spaces between arithmetic operators.

.. code:: cpp

   x = c * (a + b);  //Yes
   x = c*(a+b).      // No, the clarity is better if there are spaces between binary operators and operands.

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
~~~~~~~~~~~~~~~~

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
