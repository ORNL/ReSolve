#include "Stopwatch.hpp"
#include <iostream>

namespace ReSolve {
  Stopwatch::Stopwatch() : paused(false)
  {
    elapsed_time = std::chrono::duration<double>::zero();
  }

  /**
   * @brief Start or resume the stopwatch.
   */
  void Stopwatch::start() {
    start_time = std::chrono::steady_clock::now();
    paused = false;
  }

  /**
   * @brief Start a new lap for the stopwatch without pausing.
   */
  void Stopwatch::startLap() {
    lap_start_time = std::chrono::steady_clock::now();
  }

  /** 
   * @brief Return time between last call to start() and this call.
   *
   * @return Elapsed time in seconds since the stopwatch was started or resumed.
   */
  double Stopwatch::lapElapsed() const {
    auto current_time = std::chrono::steady_clock::now();
    auto lap_duration = current_time - lap_start_time;
    return std::chrono::duration<double>(lap_duration).count();
  }

  /**
   * @brief Pause the stopwatch.
   * 
   */
  void Stopwatch::pause() {
    if (!paused) {
      auto end_time = std::chrono::steady_clock::now();
      elapsed_time += end_time - start_time;
      paused = true;
    }
  }

  /**
  * @brief Get the elapsed time in seconds.
  * @return Elapsed time in seconds. If the stopwatch is not paused, 
  *         this includes the time since the last call to start().
  */
  double Stopwatch::totalElapsed() const {
    if (!paused) {
      auto current_time = std::chrono::steady_clock::now();
      auto total_duration = current_time - start_time;
      return (elapsed_time + total_duration).count();
    } else {
      return elapsed_time.count();
    }
  }
}

