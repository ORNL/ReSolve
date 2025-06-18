/**
 * @file LoggerTests.hpp
 * @brief Contains definition of LoggerTests class.
 * @author Slaven Peles <peless@ornl.org>
 */

#pragma once
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Class implementing unit tests for Logger class.
     *
     * The LoggerTests class is implemented entirely in this header file.
     * Adding new unit test requires simply adding another method to this
     * class.
     */
    class LoggerTests : TestBase
    {
    public:
      LoggerTests()
      {
      }

      virtual ~LoggerTests()
      {
      }

      /**
       * @brief Test data stream for error log messages.
       *
       * This method tests streaming messages to `Logger::error()` data
       * stream. The method streams messages to all available output streams,
       * however only mesages streamed to the error stream should be logged.
       */
      TestOutcome errorOutput()
      {
        using out = ReSolve::io::Logger;
        std::string s1("Test error output ...");
        std::string s2("Another error output test ...\n");
        std::string answer = error_text() + s1 + "\n" + error_text() + s2;

        TestStatus status;

        std::ostringstream file;

        out::setOutput(file);
        out::setVerbosity(out::ERRORS);
        out::error() << s1 << std::endl;
        out::error() << s2;

        out::warning() << s1;
        out::warning() << s2;
        out::summary() << s1;
        out::misc() << s1;

        // std::cout << file.str();
        // std::cout << answer;

        status = (answer == file.str());

        return status.report(__func__);
      }

      /**
       * @brief Test data stream for warning log messages.
       *
       * This method tests streaming messages to `Logger::error()` data
       * stream. The method streams messages to all available output streams,
       * however only mesages streamed to the error and warning streams should
       * be logged.
       */
      TestOutcome warningOutput()
      {
        using out = ReSolve::io::Logger;
        std::string s1("Test error output ...\n");
        std::string s2("Test warning output ...\n");
        std::string answer = error_text() + s1 + warning_text() + s2;

        TestStatus status;

        std::ostringstream file;

        out::setOutput(file);
        out::setVerbosity(out::WARNINGS);

        out::error() << s1;
        out::warning() << s2;
        out::summary() << s1;
        out::misc() << s1;

        // std::cout << file.str();

        status = (answer == file.str());

        return status.report(__func__);
      }

      /**
       * @brief Test data stream for result summary log messages.
       *
       * This method tests streaming messages to `Logger::error()` data
       * stream. The method streams messages to all available output streams,
       * however only mesages streamed to the error, warning, and result summary
       * streams should be logged.
       */
      TestOutcome summaryOutput()
      {
        using out = ReSolve::io::Logger;
        std::string s1("Test error output ...\n");
        std::string s2("Test warning output ...\n");
        std::string s3("Test summary output ...\n");
        std::string answer = error_text() + s1 + warning_text() + s2 + summary_ + s3;

        TestStatus status;

        std::ostringstream file;

        out::setOutput(file);
        out::setVerbosity(out::SUMMARY);

        out::error() << s1;
        out::warning() << s2;
        out::summary() << s3;
        out::misc() << s1;

        // std::cout << file.str();

        status = (answer == file.str());

        return status.report(__func__);
      }

      /**
       * @brief Test data stream for all other log messages.
       *
       * This method tests streaming messages to `Logger::error()` data
       * stream. The method streams messages to all available output streams
       * and all messages should be logged.
       */
      TestOutcome miscOutput()
      {
        using out = ReSolve::io::Logger;
        std::string s1("Test error output ...\n");
        std::string s2("Test warning output ...\n");
        std::string s3("Test summary output ...\n");
        std::string s4("Test any other output ...\n");
        std::string answer = error_text() + s1 + warning_text() + s2 + summary_ + s3 + message_ + s4;

        TestStatus status;

        std::ostringstream file;

        out::setOutput(file);
        out::setVerbosity(out::EVERYTHING);

        out::error() << s1;
        out::warning() << s2;
        out::summary() << s3;
        out::misc() << s4;

        // std::cout << file.str();

        status = (answer == file.str());

        return status.report(__func__);
      }

    private:
      /// Private method to return the string preceding error output
      std::string error_text()
      {
        using namespace colors;
        std::ostringstream stream;
        stream << "[" << RED << "ERROR" << CLEAR << "] ";
        return stream.str();
      }

      /// Private method to return the string preceding warning output
      std::string warning_text()
      {
        using namespace colors;
        std::ostringstream stream;
        stream << "[" << YELLOW << "WARNING" << CLEAR << "] ";
        return stream.str();
      }

      /// String preceding output of a result summary
      const std::string summary_ = "[SUMMARY] ";

      /// String preceding miscellaneous output
      const std::string message_ = "[MESSAGE] ";
    }; // class LoggerTests

  } // namespace tests
} // namespace ReSolve
