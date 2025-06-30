#include "Stopwatch.hpp"
#include <iostream>

Stopwatch::Stopwatch() : running(false), paused(false), elapsed_time(0) {}

void Stopwatch::start() {
    if (paused) {
        Stopwatch::resume();
        return;
    }

    running = true;
    paused = false;
    elapsed_time = std::chrono::duration<double>::zero();
    start_time = std::chrono::steady_clock::now();
}

void Stopwatch::pause() {
    if (running && !paused) {
        pause_time = std::chrono::steady_clock::now();
        elapsed_time += pause_time - start_time;
        paused = true;
    }
}

void Stopwatch::resume() {
    if (running && paused) {
        start_time = std::chrono::steady_clock::now();
        paused = false;
    }
}

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