#include "Stopwatch.hpp"
#include <iostream>

Stopwatch::Stopwatch() : running(false), paused(false), elapsed_time(0) {}

/**
 * @brief Start or resume the stopwatch.
 */
void Stopwatch::start() {
    if (running && paused) {
        start_time = std::chrono::steady_clock::now();
        paused = false;
        return;
    }

    running = true;
    paused = false;
    elapsed_time = std::chrono::duration<double>::zero();
    start_time = std::chrono::steady_clock::now();
}

/**
 * @brief Pause the stopwatch.
 * @return Elapsed time in seconds since the stopwatch was started or resumed.
 */
double Stopwatch::pause() {
    if (running && !paused) {
        pause_time = std::chrono::steady_clock::now();
        auto elapsed_diff = pause_time - start_time;
        elapsed_time += elapsed_diff;
        paused = true;
        return std::chrono::duration<double>(elapsed_diff).count();
    }
    return 0;
}

/**
 * @brief Stop the stopwatch.
 * If the stopwatch is running, it will record the elapsed time until now.
 */
void Stopwatch::stop() {
    if (running) {
        if (!paused) {
            auto end_time = std::chrono::steady_clock::now();
            elapsed_time += end_time - start_time;
        }
        running = false;
        paused = false;
    }
}

/**
 * @brief Get the elapsed time in seconds.
 * If the stopwatch is running, it will return the total elapsed time.
 * If stopped or paused, it will return the total elapsed time until it was stopped.
 * @return Elapsed time in seconds.
 */
double Stopwatch::elapsed() const {
    if (running) {
        if (paused) {
            return elapsed_time.count();
        } else {
            auto now = std::chrono::steady_clock::now();
            return (elapsed_time + (now - start_time)).count();
        }
    }
    return elapsed_time.count();
}