/**
 * @file 
*/

#pragma once

#include <iostream>
#include <fstream>
#include <vector>

namespace ReSolve
{
  namespace io
  {
    /**
     * @brief Class that manages and logs outputs from Re::Solve code.
     * 
     * All methods and data in this class are static.
     * 
     */
    class Logger
    {
      public:
        /// Enum specifying verbosity level for the output.
        enum Verbosity {NONE=0, ERRORS, WARNINGS, SUMMARY, EVERYTHING};

        // All methods and data are static so delete constructor and destructor.
        Logger()  = delete;
        ~Logger() = delete;

        static std::ostream& error();
        static std::ostream& warning();
        static std::ostream& summary();
        static std::ostream& misc();

        static void setOutput(std::ostream& out);
        static void openOutputFile(std::string filename);
        static void closeOutputFile();
        static void setVerbosity(Verbosity v);

        static std::vector<std::ostream*>&& init();

      private:
        static void updateVerbosity(std::vector<std::ostream*>& output_streams);

      private:
        static std::ostream nullstream_;
        static std::ofstream file_;
        static std::ostream* logger_;
        static std::vector<std::ostream*> output_streams_;
        static std::vector<std::ostream*> tmp_;
        static Verbosity verbosity_;
    };
  } // namespace io
} //namespace ReSolve