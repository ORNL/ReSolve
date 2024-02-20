#pragma once

#include <map>
#include <string>

namespace ReSolve
{

  /**
   * @brief Parser for command line input
   * 
   */
  class CliOptions
  {
  public:
      using Option = std::pair<std::string, std::string>;
      CliOptions(int argc, char *argv[]);
      virtual ~CliOptions();
      std::string getAppName() const;
      bool hasKey(const std::string&) const;
      Option* getParamFromKey(const std::string&) const;
      void printOptions() const;
  private:
      using Options = std::map<std::string, std::string>;
      void parse();
      const char* const *begin() const;
      const char* const *end() const;
      const char* const *last() const;
      Options options_;
      int argc_;
      char** argv_;
      std::string appName_;
  };

} // namespace ReSolve
