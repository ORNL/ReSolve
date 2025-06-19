#pragma once

#include <map>
#include <memory>
#include <string>

namespace ReSolve
{

  /**
   * @brief Parser for command line input
   *
   * @note Based on StackOverflow answer by Luca Davanzo
   */
  class CliOptions
  {
  public:
    using Option = std::pair<std::string, std::string>;
    CliOptions(int argc, char* argv[]);
    virtual ~CliOptions();
    std::string             getAppName() const;
    bool                    hasKey(const std::string&) const;
    std::unique_ptr<Option> getParamFromKey(const std::string&) const;
    void                    printOptionsList() const;

  private:
    using OptionsList = std::map<std::string, std::string>;
    void               parse();
    const char* const* begin() const;
    const char* const* end() const;
    const char* const* last() const;
    OptionsList        options_;
    int                argc_;
    char**             argv_;
    std::string        app_name_;
  };

} // namespace ReSolve
