/**
 * @file Logger.cpp
 * @brief Contains definition of Logger class.
 * @author Slaven Peles <peless@ornl.org>
 */

#include "Logger.hpp"

#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace io
  {
    /// @brief Default verbosity is to print error and warning messages
    Logger::Verbosity Logger::verbosity_ = Logger::WARNINGS;

    /// @brief Default output is standard output
    std::ostream* Logger::logger_ = &std::cout;

    /// @brief User provided output file stream
    std::ofstream Logger::file_;

    /// @brief Stream to null device
    std::ostream Logger::nullstream_(nullptr);

    /// @brief Auxiliary vector of output streams
    std::vector<std::ostream*> Logger::tmp_;

    /// @brief Vector of different output streams
    std::vector<std::ostream*> Logger::output_streams_(Logger::init());

    /**
     * @brief Sets verbosity level
     *
     * @pre `output_streams_` vector is allocated
     * @post Verbosity level is set to user supplied value `v` and outputs
     * for `output_streams_` are set accordingly.
     */
    void Logger::setVerbosity(Verbosity v)
    {
      verbosity_ = v;
      updateVerbosity(output_streams_);
    }

    /// @brief Gets verbosity level
    Logger::Verbosity Logger::verbosity()
    {
      return verbosity_;
    }

    /**
     * @brief Private method to update verbosity.
     *
     * This function directs each output stream <= `verbosity_` to user
     * selected output and sets all others to null device. Each output stream
     * corresponds to different verbosity level.
     *
     * @param[in] output_streams - vector of pointers to output streams
     *
     * @pre Vector `output_streams` is allocated and correctly initialized.
     * @post All streams `output_stream_[i]`, where `i <= verbosity_` are
     * directed to stream `logger_`. The rest are sent to null device
     * (not printed).
     */
    void Logger::updateVerbosity(std::vector<std::ostream*>& output_streams)
    {
      for (std::size_t i = NONE; i <= EVERYTHING; ++i)
      {
        output_streams[i] = i > verbosity_ ? &nullstream_ : logger_;
      }
    }

    /**
     * @brief Delivers default values for output streams.
     */
    std::vector<std::ostream*>& Logger::init()
    {
      tmp_.resize(Logger::EVERYTHING + 1);
      updateVerbosity(tmp_);
      return tmp_;
    }

    /**
     * @brief Returns reference to output stream for error messages.
     *
     * @return Reference to error messages stream in `output_streams_`.
     *
     * @pre `output_streams_` vector is allocated and correctly initialized.
     */
    std::ostream& Logger::error()
    {
      using namespace colors;
      *(output_streams_[ERRORS]) << "[" << RED << "ERROR" << CLEAR << "] ";
      return *(output_streams_[ERRORS]);
    }

    /**
     * @brief Returns reference to output stream for warning messages.
     *
     * @return Reference to warning messages stream in `output_streams_`.
     *
     * @pre `output_streams_` vector is allocated and correctly initialized.
     */
    std::ostream& Logger::warning()
    {
      using namespace colors;
      *(output_streams_[WARNINGS]) << "[" << YELLOW << "WARNING" << CLEAR << "] ";
      return *(output_streams_[WARNINGS]);
    }

    /**
     * @brief Returns reference to analysis summary messages output stream.
     *
     * @return Reference to analysis summary messages stream in `output_streams_`.
     *
     * @pre `output_streams_` vector is allocated and correctly initialized.
     */
    std::ostream& Logger::summary()
    {
      *(output_streams_[SUMMARY]) << "[SUMMARY] ";
      return *(output_streams_[SUMMARY]);
    }

    /**
     * @brief Returns reference to output stream for all other messages.
     *
     * @return Reference to output stream to miscellaneous messages
     * in `output_streams_`.
     *
     * @pre `output_streams_` vector is allocated and correctly initialized.
     */
    std::ostream& Logger::misc()
    {
      *(output_streams_[EVERYTHING]) << "[MESSAGE] ";
      return *(output_streams_[EVERYTHING]);
    }

    /**
     * @brief Open file `filename` and update outputs for different verbosities
     * streams.
     *
     * @param[in] filename - The name of the output file.
     *
     * @pre `output_streams_` vector is allocated and correctly initialized.
     * @post All active streams are directed to user supplied file `filename`.
     */
    void Logger::openOutputFile(std::string filename)
    {
      file_.open(filename);
      logger_ = &file_;
      updateVerbosity(output_streams_);
    }

    /**
     * @brief Set outputs of active streams to user provided `std::ostream` object.
     *
     * All active outputs are redirected to `out` stream. All inactive ones are
     * directed to null device.
     *
     * @param[in] out - User provided output stream.
     *
     * @pre `output_streams_` vector is allocated and correctly initialized.
     * @post All active streams (`output_streams_[i]` where `i <= verbosity_`)
     * are set to user provided `out` output stream.
     */
    void Logger::setOutput(std::ostream& out)
    {
      logger_ = &out;
      updateVerbosity(output_streams_);
    }

    /**
     * @brief Close output file.
     *
     * @pre Output file `file_` has been opened.
     * @post Output file `file_` is closed and active output streams are
     * set to default output `std::cout`.
     */
    void Logger::closeOutputFile()
    {
      file_.close();
      logger_ = &std::cout;
      updateVerbosity(output_streams_);
    }

  } // namespace io
} // namespace ReSolve
