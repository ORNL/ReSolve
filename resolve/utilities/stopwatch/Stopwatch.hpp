#pragma once
#include <chrono>

namespace ReSolve
{
  /**
  * @class Stopwatch
  * This class provides timing functionality for testing.
  */
  class Stopwatch {
  public:
    Stopwatch();

    void start();
    void pause();
    double lap() const;
    double totalElapsed() const;

  private:
    bool paused;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::duration<double> elapsed_time;
  };
}