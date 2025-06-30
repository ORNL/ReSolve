#pragma once
#include <chrono>

/**
 * Stopwatch class for timing operations.
 * This class allows starting, pausing, resuming, and stopping a stopwatch,
 * and provides the elapsed time in seconds.
 */
class Stopwatch {
public:
    Stopwatch();

    void start();
    double pause();
    void stop();
    double elapsed() const;

private:
    bool running;
    bool paused;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point pause_time;
    std::chrono::duration<double> elapsed_time;
};
