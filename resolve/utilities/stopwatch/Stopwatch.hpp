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
    void startLap();
    double lapElapsed() const;
    double totalElapsed() const;

  private:
    bool paused;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point lap_start_time;
    std::chrono::duration<double> elapsed_time;
  };
}