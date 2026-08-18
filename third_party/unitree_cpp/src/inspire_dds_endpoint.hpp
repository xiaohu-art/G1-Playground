#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

constexpr std::size_t kInspireSlots = 12;

struct InspireDdsEndpointConfig {
    std::int32_t domain_id = 1;
    std::string net_if = "lo";
    std::string cmd_topic = "rt/inspire/cmd";
    std::string state_topic = "rt/inspire/state";
};

struct InspireStateSnapshot {
    bool valid = false;
    std::uint64_t sequence = 0;
    double age_seconds = 0.0;
    std::vector<double> q = std::vector<double>(kInspireSlots, 1.0);
    std::vector<double> dq = std::vector<double>(kInspireSlots, 0.0);
};

struct InspireCommandSnapshot {
    bool valid = false;
    std::uint64_t sequence = 0;
    double age_seconds = 0.0;
    std::vector<double> q = std::vector<double>(kInspireSlots, 1.0);
};

class InspireDdsControlEndpoint {
   public:
    explicit InspireDdsControlEndpoint(const InspireDdsEndpointConfig& cfg);
    ~InspireDdsControlEndpoint();

    InspireStateSnapshot get_state() const;
    bool self_check() const;
    std::uint64_t step(const std::vector<double>& stroke);
    bool close();

   private:
    using MotorCmds = unitree_go::msg::dds_::MotorCmds_;
    using MotorStates = unitree_go::msg::dds_::MotorStates_;
    using Clock = std::chrono::steady_clock;

    InspireDdsEndpointConfig cfg_;
    mutable std::mutex mutex_;
    InspireStateSnapshot state_;
    Clock::time_point last_state_time_{};
    std::uint64_t published_ = 0;
    bool closed_ = false;

    unitree::robot::ChannelSubscriberPtr<MotorStates> state_subscriber_;
    unitree::robot::ChannelPublisherPtr<MotorCmds> cmd_publisher_;

    void StateHandler(const void* message);
    void CloseTransportNoexcept() noexcept;
};

class InspireDdsRobotEndpoint {
   public:
    explicit InspireDdsRobotEndpoint(const InspireDdsEndpointConfig& cfg);
    ~InspireDdsRobotEndpoint();

    InspireCommandSnapshot get_command() const;
    std::uint64_t publish_state(const std::vector<double>& stroke, const std::vector<double>& stroke_rate);
    bool close();

   private:
    using MotorCmds = unitree_go::msg::dds_::MotorCmds_;
    using MotorStates = unitree_go::msg::dds_::MotorStates_;
    using Clock = std::chrono::steady_clock;

    InspireDdsEndpointConfig cfg_;
    mutable std::mutex mutex_;
    InspireCommandSnapshot command_;
    Clock::time_point last_command_time_{};
    std::uint64_t published_ = 0;
    bool closed_ = false;

    unitree::robot::ChannelSubscriberPtr<MotorCmds> cmd_subscriber_;
    unitree::robot::ChannelPublisherPtr<MotorStates> state_publisher_;

    void CommandHandler(const void* message);
    void CloseTransportNoexcept() noexcept;
};
