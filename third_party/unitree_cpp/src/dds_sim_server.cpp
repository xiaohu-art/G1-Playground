#include "dds_sim_server.hpp"
#include "unitree_controller.hpp"

#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

#include <unitree/robot/channel/channel_factory.hpp>

using unitree::robot::ChannelFactory;

namespace {

constexpr std::size_t kNumDofs = 29;

template <typename Message>
std::uint32_t MessageCrc(Message& message) {
    return unitree_cpp_detail::Crc32Core(
        reinterpret_cast<const std::uint32_t*>(&message), (sizeof(Message) >> 2) - 1);
}

void ValidateVector(const std::vector<double>& values, const char* name) {
    if (values.size() != kNumDofs) {
        throw std::invalid_argument(std::string(name) + " must contain 29 values");
    }
    for (double value : values) {
        if (!std::isfinite(value) || std::abs(value) > FLT_MAX) {
            throw std::invalid_argument(std::string(name) + " must contain values representable as floats");
        }
    }
}

template <std::size_t Size>
void ValidateArray(const std::array<double, Size>& values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value) || std::abs(value) > FLT_MAX) {
            throw std::invalid_argument(std::string(name) + " must contain values representable as floats");
        }
    }
}

}  // namespace

G1DdsSimServer::G1DdsSimServer(const DdsSimServerConfig& cfg) : cfg_(cfg) {
    if (cfg_.domain_id < 0 || cfg_.net_if.empty()) {
        throw std::invalid_argument("DDS simulator endpoint is invalid");
    }
    if (cfg_.mode_machine != 5) {
        throw std::invalid_argument("G1 29-DoF DDS simulator requires mode_machine=5");
    }

    unitree_cpp_detail::InitializeDdsEndpointOnce(cfg_.domain_id, cfg_.net_if);
    try {
        lowcmd_subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<LowCmd>>(cfg_.lowcmd_topic);
        lowcmd_subscriber_->InitChannel(
            [this](const void* message) { LowCommandHandler(message); }, 1);
        lowstate_publisher_ = std::make_shared<unitree::robot::ChannelPublisher<LowState>>(cfg_.lowstate_topic);
        lowstate_publisher_->InitChannel();
    } catch (...) {
        CloseTransportNoexcept();
        throw;
    }
}

G1DdsSimServer::~G1DdsSimServer() {
    CloseTransportNoexcept();
}

void G1DdsSimServer::LowCommandHandler(const void* message) {
    LowCmd low_command = *static_cast<const LowCmd*>(message);
    DdsCommandSnapshot accepted;
    const auto validation = unitree_cpp_detail::ValidateDdsCommandForTest(low_command, cfg_.mode_machine, &accepted);

    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    if (validation == unitree_cpp_detail::DdsCommandValidation::CRC_ERROR) {
        ++stats_.crc_errors;
        return;
    }
    if (validation == unitree_cpp_detail::DdsCommandValidation::MODE_ERROR) {
        ++stats_.mode_errors;
        return;
    }
    if (validation == unitree_cpp_detail::DdsCommandValidation::VALUE_ERROR) {
        ++stats_.finite_errors;
        return;
    }
    accepted.sequence = ++stats_.accepted_commands;
    command_ = std::move(accepted);
    last_command_time_ = Clock::now();
}

DdsCommandSnapshot G1DdsSimServer::get_command() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        throw std::logic_error("DDS simulator server is closed");
    }
    DdsCommandSnapshot snapshot = command_;
    snapshot.age_seconds = snapshot.valid
                               ? std::chrono::duration<double>(Clock::now() - last_command_time_).count()
                               : std::numeric_limits<double>::infinity();
    return snapshot;
}

std::uint32_t G1DdsSimServer::publish_lowstate(const DdsLowStateSnapshot& snapshot) {
    ValidateVector(snapshot.q, "q");
    ValidateVector(snapshot.dq, "dq");
    ValidateVector(snapshot.tau_est, "tau_est");
    ValidateArray(snapshot.quaternion, "quaternion");
    ValidateArray(snapshot.gyroscope, "gyroscope");
    ValidateArray(snapshot.accelerometer, "accelerometer");
    ValidateArray(snapshot.rpy, "rpy");

    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || !lowstate_publisher_) {
        throw std::logic_error("DDS simulator server is closed");
    }

    unitree_hg::msg::dds_::LowState_ low_state{};
    unitree_cpp_detail::FillDdsLowStateForTest(snapshot, cfg_.mode_machine, next_tick_, &low_state);
    if (!lowstate_publisher_->Write(low_state)) {
        throw std::runtime_error("Failed to publish DDS LowState");
    }
    return next_tick_++;
}

DdsSimServerStats G1DdsSimServer::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

bool G1DdsSimServer::close() {
    unitree::robot::ChannelSubscriberPtr<LowCmd> subscriber;
    unitree::robot::ChannelPublisherPtr<LowState> publisher;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return false;
        }
        closed_ = true;
        subscriber = std::move(lowcmd_subscriber_);
        publisher = std::move(lowstate_publisher_);
    }
    if (subscriber) {
        subscriber->CloseChannel();
    }
    if (publisher) {
        publisher->CloseChannel();
    }
    return true;
}

void G1DdsSimServer::CloseTransportNoexcept() noexcept {
    try {
        close();
    } catch (...) {
    }
}

unitree_cpp_detail::DdsCommandValidation unitree_cpp_detail::ValidateDdsCommandForTest(
    unitree_hg::msg::dds_::LowCmd_ command, std::uint8_t expected_mode_machine, DdsCommandSnapshot* snapshot) {
    if (snapshot == nullptr) {
        throw std::invalid_argument("command snapshot output must not be null");
    }
    if (command.crc() != MessageCrc(command)) {
        return DdsCommandValidation::CRC_ERROR;
    }
    if (command.mode_pr() != 0 || command.mode_machine() != expected_mode_machine) {
        return DdsCommandValidation::MODE_ERROR;
    }

    DdsCommandSnapshot accepted;
    accepted.valid = true;
    accepted.mode_pr = command.mode_pr();
    accepted.mode_machine = command.mode_machine();
    for (std::size_t index = 0; index < kNumDofs; ++index) {
        const auto& motor = command.motor_cmd().at(index);
        if (motor.mode() != 1 || !std::isfinite(motor.q()) || !std::isfinite(motor.dq()) ||
            !std::isfinite(motor.tau()) || !std::isfinite(motor.kp()) || !std::isfinite(motor.kd())) {
            return DdsCommandValidation::VALUE_ERROR;
        }
        accepted.q[index] = motor.q();
        accepted.dq[index] = motor.dq();
        accepted.tau[index] = motor.tau();
        accepted.kp[index] = motor.kp();
        accepted.kd[index] = motor.kd();
    }
    *snapshot = std::move(accepted);
    return DdsCommandValidation::ACCEPTED;
}

std::uint32_t unitree_cpp_detail::FillDdsLowStateForTest(
    const DdsLowStateSnapshot& snapshot, std::uint8_t mode_machine, std::uint32_t tick,
    unitree_hg::msg::dds_::LowState_* message) {
    if (message == nullptr) {
        throw std::invalid_argument("LowState output must not be null");
    }
    ValidateVector(snapshot.q, "q");
    ValidateVector(snapshot.dq, "dq");
    ValidateVector(snapshot.tau_est, "tau_est");
    ValidateArray(snapshot.quaternion, "quaternion");
    ValidateArray(snapshot.gyroscope, "gyroscope");
    ValidateArray(snapshot.accelerometer, "accelerometer");
    ValidateArray(snapshot.rpy, "rpy");

    unitree_hg::msg::dds_::LowState_ low_state{};
    low_state.mode_pr(0);
    low_state.mode_machine(mode_machine);
    low_state.tick(tick);
    for (std::size_t index = 0; index < kNumDofs; ++index) {
        auto& motor = low_state.motor_state().at(index);
        motor.mode(1);
        motor.q(static_cast<float>(snapshot.q[index]));
        motor.dq(static_cast<float>(snapshot.dq[index]));
        motor.tau_est(static_cast<float>(snapshot.tau_est[index]));
    }
    auto& imu = low_state.imu_state();
    for (std::size_t index = 0; index < 4; ++index) {
        imu.quaternion().at(index) = static_cast<float>(snapshot.quaternion[index]);
    }
    for (std::size_t index = 0; index < 3; ++index) {
        imu.gyroscope().at(index) = static_cast<float>(snapshot.gyroscope[index]);
        imu.accelerometer().at(index) = static_cast<float>(snapshot.accelerometer[index]);
        imu.rpy().at(index) = static_cast<float>(snapshot.rpy[index]);
    }
    low_state.wireless_remote() = snapshot.wireless_remote;
    low_state.crc(MessageCrc(low_state));
    *message = std::move(low_state);
    return message->crc();
}
