#include <iostream>

#include "CliOptions.hpp"


namespace ReSolve
{

  CliOptions::CliOptions(int argc, char* argv[])
    : argc_(argc),
      argv_(argv)
  {
    appName_ = argv_[0];
    parse();
  }

  CliOptions::~CliOptions()
  {
  }

  std::string CliOptions::getAppName() const
  {
    return appName_;
  }

  bool CliOptions::hasKey(const std::string& key) const
  {
    return options_.find(key) != options_.end();
  }

  CliOptions::Option* CliOptions::getParamFromKey(const std::string& key) const
  {
    const Options::const_iterator i = options_.find(key);
    CliOptions::Option* opt = 0;
    if (i != options_.end()) {
      opt = new CliOptions::Option((*i).first, (*i).second);
    }
    return opt;
  }

  void CliOptions::printOptions() const
  {
    Options::const_iterator m = options_.begin();
    int i = 0;
    if (options_.empty()) {
        std::cout << "No parameters\n";
    }
    for (; m != options_.end(); m++, ++i) {
      std::cout << "Parameter [" << i << "] [" 
                << (*m).first  << " " 
                << (*m).second << "]\n";
    }
  }

  //
  // Private methods
  //

  /**
   * @brief Parse command line input and store it in a map
   * 
   */
  void CliOptions::parse()
  {
    Option* option = new std::pair<std::string, std::string>();
    // Loop over argv entries skipping the first one (executable name)
    for (const char* const* i = this->begin() + 1; i != this->end(); i++)
    {
      const std::string p = *i;
      if (option->first == "" && p[0] == '-')
      {
        // Set option ID
        option->first = p;
        if (i == this->last())
        {
          // If this is last entry, there is nothing else to do; set option.
          options_.insert(Option(option->first, option->second));
        }
        continue;
      } 
      else if (option->first != "" && p[0] == '-')
      {
        // Option ID has been set in prior cycle, string p is also option ID.
        option->second = "null"; /* or leave empty? */
        // Set option without parameter value
        options_.insert(Option(option->first, option->second));
        // Set parameter ID for the next option
        option->first = p;
        option->second = "";
        if (i == this->last())
        {
          // If this is last entry, there is nothing else to do; set option.
          options_.insert(Option(option->first, option->second));
        }
        continue;
      }
      else if (option->first != "")
      {
        // String p contains parameter value
        option->second = p;
        // Set option with parameter value
        options_.insert(Option(option->first, option->second));
        // Reset option to receive the next entry
        option->first = "";
        option->second = "";
        continue;
      }
    }
  }

  const char* const *CliOptions::begin() const
  {
      return argv_;
  }

  const char* const *CliOptions::end() const
  {
      return argv_ + argc_;
  }

  const char* const *CliOptions::last() const
  {
      return argv_ + argc_ - 1;
  }


} // namespace ReSolve
