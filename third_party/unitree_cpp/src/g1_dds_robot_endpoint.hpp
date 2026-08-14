#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

struct G1DdsRobotEndpointConfig {
    std::int32_t domain_id = 1;
    std::string net_if = "lo";
    std::string lowcmd_topic = "rt/lowcmd";
    std::string lowstate_topic = "rt/lowstate";
    std::uint8_t mode_machine = 5;
};

struct DdsCommandSnapshot {
    bool valid = false;
    std::uint64_t sequence = 0;
    double age_seconds = 0.0;
    std::uint8_t mode_pr = 0;
    std::uint8_t mode_machine = 0;
    std::vector<double> q = std::vector<double>(29, 0.0);
    std::vector<double> dq = std::vector<double>(29, 0.0);
    std::vector<double> tau = std::vector<double>(29, 0.0);
    std::vector<double> kp = std::vector<double>(29, 0.0);
    std::vector<double> kd = std::vector<double>(29, 0.0);
};

struct DdsLowStateSnapshot {
    std::vector<double> q = std::vector<double>(29, 0.0);
    std::vector<double> dq = std::vector<double>(29, 0.0);
    std::vector<double> tau_est = std::vector<double>(29, 0.0);
    std::array<double, 4> quaternion = {1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> gyroscope = {0.0, 0.0, 0.0};
    std::array<double, 3> accelerometer = {0.0, 0.0, 0.0};
    std::array<double, 3> rpy = {0.0, 0.0, 0.0};
    std::array<std::uint8_t, 40> wireless_remote = {};
};

struct G1DdsRobotEndpointStats {
    std::uint64_t accepted_commands = 0;
    std::uint64_t crc_errors = 0;
    std::uint64_t finite_errors = 0;
    std::uint64_t mode_errors = 0;
};

class G1DdsRobotEndpoint {
   public:
    explicit G1DdsRobotEndpoint(const G1DdsRobotEndpointConfig& cfg);
    ~G1DdsRobotEndpoint();

    DdsCommandSnapshot get_command() const;
    std::uint32_t publish_lowstate(const DdsLowStateSnapshot& snapshot);
    G1DdsRobotEndpointStats stats() const;
    bool close();

   private:
    using LowCmd = unitree_hg::msg::dds_::LowCmd_;
    using LowState = unitree_hg::msg::dds_::LowState_;
    using Clock = std::chrono::steady_clock;

    G1DdsRobotEndpointConfig cfg_;
    mutable std::mutex mutex_;
    DdsCommandSnapshot command_;
    G1DdsRobotEndpointStats stats_;
    Clock::time_point last_command_time_{};
    std::uint32_t next_tick_ = 1;
    bool closed_ = false;

    unitree::robot::ChannelSubscriberPtr<LowCmd> lowcmd_subscriber_;
    unitree::robot::ChannelPublisherPtr<LowState> lowstate_publisher_;

    void LowCommandHandler(const void* message);
    void CloseTransportNoexcept() noexcept;
};

namespace unitree_cpp_detail {

enum class DdsCommandValidation {
    ACCEPTED = 0,
    CRC_ERROR = 1,
    MODE_ERROR = 2,
    VALUE_ERROR = 3,
};

DdsCommandValidation ValidateDdsCommandForTest(
    unitree_hg::msg::dds_::LowCmd_ command, std::uint8_t expected_mode_machine, DdsCommandSnapshot* snapshot);
std::uint32_t FillDdsLowStateForTest(
    const DdsLowStateSnapshot& snapshot, std::uint8_t mode_machine, std::uint32_t tick,
    unitree_hg::msg::dds_::LowState_* message);

}  // namespace unitree_cpp_detail
