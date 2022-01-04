#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace gtn {
namespace detail {

  // Step 1: enqueu all the work in different threads
  // Step 2: Each thread records its event
  // Step 3: Wait for threads to record
  // Step 4: Each thread waits on all other threads
class CudaThreadPool {
 public:
  CudaThreadPool(size_t threads) : stop(false) {
    tasks.resize(threads);
    events.resize(threads);
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back([this, i] {
        auto& taskQ = tasks[i];
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(
                lock, [this, &taskQ] { return stop || !taskQ.empty(); });
            if (stop && taskQ.empty())
              return;
            task = std::move(taskQ.front());
            taskQ.pop();
          }
          task();
        }
      });
  }

  void syncThreads() {
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
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks[i].push(task);
    }
    condition.notify_all();
  }

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    cuda::synchronizeStream();
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);

      // don't allow enqueueing after stopping the pool
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      tasks[taskCounter_].emplace([task]() { (*task)(); });
      taskCounter_ = (taskCounter_ + 1) % tasks.size();
    }
    condition.notify_all();
    return res;
  }

  ~CudaThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers)
      worker.join();
  }

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queues
  std::vector<std::queue<std::function<void()>>> tasks;
  int taskCounter_{0};

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::vector<cuda::Event> events;
  bool stop;
};

} // namespace detail
} // namespace gtn
