#ifndef HAND_WORKER_H
#define HAND_WORKER_H

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <eigen3/Eigen/Dense>

#include "inspire.h"

class HandWorker
{
public:
  using Vector6 = Eigen::Matrix<double, 6, 1>;
  using SharedPtr = std::shared_ptr<HandWorker>;

  static constexpr double VELOCITY_ALPHA = 0.3;
  static constexpr double MIN_VELOCITY_DT = 1e-4;
  static constexpr double MAX_VELOCITY_DT = 0.5;

  HandWorker(std::shared_ptr<inspire::InspireHand> hand, std::string name)
  : hand_(std::move(hand)), name_(std::move(name))
  {
    target_.setZero();
    state_.setZero();
    velocity_.setZero();
  }

  ~HandWorker() { stop(); }

  HandWorker(const HandWorker &) = delete;
  HandWorker & operator=(const HandWorker &) = delete;

  void start()
  {
    if (running_.exchange(true)) return;
    thread_ = std::thread(&HandWorker::loop, this);
  }

  void stop()
  {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
  }

  void set_target(const Vector6 & q)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    target_ = q;
    have_target_ = true;
  }

  void set_command_active(bool active) { command_active_.store(active); }

  bool get_state(Vector6 & q, Vector6 & dq, uint32_t & lost) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    q = state_;
    dq = velocity_;
    lost = lost_;
    return have_state_;
  }

  const std::string & name() const { return name_; }
  uint64_t transactions() const { return transactions_.load(); }

private:
  void loop()
  {
    while (running_.load())
    {
      Vector6 target;
      bool send = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        target = target_;
        send = have_target_ && command_active_.load();
      }

      if (send)
      {
        hand_->SetPosition(target);
        transactions_.fetch_add(1);
      }

      Vector6 q;
      if (hand_->GetPosition(q) == 0)
      {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        if (have_state_)
        {
          const double dt = std::chrono::duration<double>(now - state_time_).count();
          if (dt > MIN_VELOCITY_DT && dt < MAX_VELOCITY_DT)
          {
            velocity_ = VELOCITY_ALPHA * ((q - state_) / dt) + (1.0 - VELOCITY_ALPHA) * velocity_;
          }
          else
          {
            velocity_.setZero();
          }
        }
        else
        {
          velocity_.setZero();
        }
        state_ = q;
        state_time_ = now;
        have_state_ = true;
      }
      else
      {
        std::lock_guard<std::mutex> lock(mutex_);
        lost_++;
      }
      transactions_.fetch_add(1);
      cycles_.fetch_add(1);
    }
  }

public:
  uint64_t cycles() const { return cycles_.load(); }

private:
  std::shared_ptr<inspire::InspireHand> hand_;
  std::string name_;
  std::thread thread_;

  std::atomic<bool> running_{false};
  std::atomic<bool> command_active_{false};
  std::atomic<uint64_t> cycles_{0};
  std::atomic<uint64_t> transactions_{0};

  mutable std::mutex mutex_;
  Vector6 target_;
  Vector6 state_;
  Vector6 velocity_;
  std::chrono::steady_clock::time_point state_time_;
  bool have_target_{false};
  bool have_state_{false};
  uint32_t lost_{0};
};

#endif // HAND_WORKER_H
