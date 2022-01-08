/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gtn/cuda/cuda.h"

namespace gtn {
namespace detail {
/**
 * A simple thread pool implementation from
 * https://github.com/progschj/ThreadPool for use in benchmarking
 * batch-parallelism across threads.
 */
class ThreadPool {
 public:
  ThreadPool(size_t threads) : stop(false) {
    tasks.resize(threads);
    if (cuda::isAvailable()) {
      events.resize(threads);
    }
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back([this, i] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(
              lock, [this, i] {
                return stop || !tasks[i].empty() || !sharedTasks.empty();
              });
            if (stop && tasks[i].empty() && sharedTasks.empty()) {
              if (cuda::isAvailable()) {
                cuda::synchronizeStream();
              }
              return;
            }
            if (!tasks[i].empty()) {
              task = std::move(tasks[i].front());
              tasks[i].pop();
            } else {
              task = std::move(sharedTasks.front());
              sharedTasks.pop();
            }
          }
          task();
        }
      });
  }
  void syncStreams() {
    // Every thread records an event, then the stream of each
    // thread waits on all events
    std::vector<std::future<void>> results;
    for (int i = 0; i < events.size(); i++) {
      auto func = [&event = events[i]]() { event.record(); };
      auto task = std::make_shared<std::packaged_task<void()>>(func);
      results.push_back(task->get_future());
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks[i].emplace([task]() { (*task)(); });
    }
    condition.notify_all();

    for (auto& f : results) {
      f.get();
    }

    for (int i = 0; i < events.size(); i++) {
      auto task = [this]() {
        for (auto& event : events) {
          event.wait();
        }
      };
      // Wait on events in the main stream as well
      task();
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks[i].push(task);
    }
    condition.notify_all();
  }

  template <class F, class... Args>
  auto enqueueIndex(int taskNum, F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    if (cuda::isAvailable()) {
      cuda::synchronizeStream();
    }
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);

      // don't allow enqueueing after stopping the pool
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      if (taskNum >= 0) {
        auto tIdx = taskNum % tasks.size();
        tasks[tIdx].emplace([task]() { (*task)(); });
      } else {
        sharedTasks.emplace([task]() { (*task)(); });
      }
    }
    condition.notify_all();
    return res;
  }

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    return enqueueIndex(-1, std::forward<F>(f), std::forward<Args>(args)...);
  }
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queues
  std::queue<std::function<void()>> sharedTasks;
  std::vector<std::queue<std::function<void()>>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::vector<cuda::Event> events;
  bool stop;
};

} // namespace detail
} // namespace gtn
