#include "inspire_dds_endpoint.hpp"
#include "dds_utils.hpp"

#include <cfloat>
#include <cmath>
#include <stdexcept>

#include <unitree/robot/channel/channel_factory.hpp>

namespace {

void ValidateStroke(const std::vector<double>& values, const char* name) {
    if (values.size() != kInspireSlots) {
        throw std::invalid_argument(std::string(name) + " must contain 12 values");
    }
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            throw std::invalid_argument(std::string(name) + " must contain finite values within [0, 1]");
        }
    }
}

void ValidateRate(const std::vector<double>& values, const char* name) {
    if (values.size() != kInspireSlots) {
        throw std::invalid_argument(std::string(name) + " must contain 12 values");
    }
    for (double value : values) {
        if (!std::isfinite(value) || std::abs(value) > FLT_MAX) {
            throw std::invalid_argument(std::string(name) + " must contain values representable as floats");
        }
    }
}

}  // namespace

InspireDdsControlEndpoint::InspireDdsControlEndpoint(const InspireDdsEndpointConfig& cfg) : cfg_(cfg) {
    if (cfg_.domain_id < 0 || cfg_.net_if.empty()) {
        throw std::invalid_argument("Inspire DDS control endpoint is invalid");
    }

    unitree_cpp_detail::InitializeDdsEndpointOnce(cfg_.domain_id, cfg_.net_if);
    try {
        state_subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<MotorStates>>(cfg_.state_topic);
        state_subscriber_->InitChannel([this](const void* message) { StateHandler(message); }, 1);
        cmd_publisher_ = std::make_shared<unitree::robot::ChannelPublisher<MotorCmds>>(cfg_.cmd_topic);
        cmd_publisher_->InitChannel();
    } catch (...) {
        CloseTransportNoexcept();
        throw;
    }
}

InspireDdsControlEndpoint::~InspireDdsControlEndpoint() {
    CloseTransportNoexcept();
}

void InspireDdsControlEndpoint::StateHandler(const void* message) {
    const MotorStates& states = *static_cast<const MotorStates*>(message);
    if (states.states().size() != kInspireSlots) {
        return;
    }

    InspireStateSnapshot snapshot;
    for (std::size_t index = 0; index < kInspireSlots; ++index) {
        const double q = static_cast<double>(states.states()[index].q());
        const double dq = static_cast<double>(states.states()[index].dq());
        if (!std::isfinite(q) || !std::isfinite(dq)) {
            return;
        }
        snapshot.q[index] = q;
        snapshot.dq[index] = dq;
        snapshot.lost[index] = states.states()[index].lost();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    snapshot.valid = true;
    snapshot.sequence = state_.sequence + 1;
    state_ = std::move(snapshot);
    last_state_time_ = Clock::now();
}

InspireStateSnapshot InspireDdsControlEndpoint::get_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    InspireStateSnapshot snapshot = state_;
    if (snapshot.valid) {
        snapshot.age_seconds = std::chrono::duration<double>(Clock::now() - last_state_time_).count();
    }
    return snapshot;
}

bool InspireDdsControlEndpoint::self_check() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_.valid;
}

std::uint64_t InspireDdsControlEndpoint::step(const std::vector<double>& stroke) {
    ValidateStroke(stroke, "Inspire stroke command");

    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || !cmd_publisher_) {
        throw std::runtime_error("Inspire DDS control endpoint is closed");
    }

    MotorCmds message;
    message.cmds().resize(kInspireSlots);
    for (std::size_t index = 0; index < kInspireSlots; ++index) {
        message.cmds()[index].q(static_cast<float>(stroke[index]));
    }
    cmd_publisher_->Write(message);
    return ++published_;
}

bool InspireDdsControlEndpoint::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return false;
    }
    closed_ = true;
    CloseTransportNoexcept();
    return true;
}

void InspireDdsControlEndpoint::CloseTransportNoexcept() noexcept {
    state_subscriber_.reset();
    cmd_publisher_.reset();
}

InspireDdsRobotEndpoint::InspireDdsRobotEndpoint(const InspireDdsEndpointConfig& cfg) : cfg_(cfg) {
    if (cfg_.domain_id < 0 || cfg_.net_if.empty()) {
        throw std::invalid_argument("Inspire DDS robot endpoint is invalid");
    }

    unitree_cpp_detail::InitializeDdsEndpointOnce(cfg_.domain_id, cfg_.net_if);
    try {
        cmd_subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<MotorCmds>>(cfg_.cmd_topic);
        cmd_subscriber_->InitChannel([this](const void* message) { CommandHandler(message); }, 1);
        state_publisher_ = std::make_shared<unitree::robot::ChannelPublisher<MotorStates>>(cfg_.state_topic);
        state_publisher_->InitChannel();
    } catch (...) {
        CloseTransportNoexcept();
        throw;
    }
}

InspireDdsRobotEndpoint::~InspireDdsRobotEndpoint() {
    CloseTransportNoexcept();
}

void InspireDdsRobotEndpoint::CommandHandler(const void* message) {
    const MotorCmds& commands = *static_cast<const MotorCmds*>(message);
    if (commands.cmds().size() != kInspireSlots) {
        return;
    }

    InspireCommandSnapshot snapshot;
    for (std::size_t index = 0; index < kInspireSlots; ++index) {
        const double q = static_cast<double>(commands.cmds()[index].q());
        if (!std::isfinite(q)) {
            return;
        }
        snapshot.q[index] = q < 0.0 ? 0.0 : (q > 1.0 ? 1.0 : q);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    snapshot.valid = true;
    snapshot.sequence = command_.sequence + 1;
    command_ = std::move(snapshot);
    last_command_time_ = Clock::now();
}

InspireCommandSnapshot InspireDdsRobotEndpoint::get_command() const {
    std::lock_guard<std::mutex> lock(mutex_);
    InspireCommandSnapshot snapshot = command_;
    if (snapshot.valid) {
        snapshot.age_seconds = std::chrono::duration<double>(Clock::now() - last_command_time_).count();
    }
    return snapshot;
}

std::uint64_t InspireDdsRobotEndpoint::publish_state(
    const std::vector<double>& stroke, const std::vector<double>& stroke_rate) {
    ValidateStroke(stroke, "Inspire stroke state");
    ValidateRate(stroke_rate, "Inspire stroke rate");

    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || !state_publisher_) {
        throw std::runtime_error("Inspire DDS robot endpoint is closed");
    }

    MotorStates message;
    message.states().resize(kInspireSlots);
    for (std::size_t index = 0; index < kInspireSlots; ++index) {
        message.states()[index].q(static_cast<float>(stroke[index]));
        message.states()[index].dq(static_cast<float>(stroke_rate[index]));
    }
    state_publisher_->Write(message);
    return ++published_;
}

bool InspireDdsRobotEndpoint::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return false;
    }
    closed_ = true;
    CloseTransportNoexcept();
    return true;
}

void InspireDdsRobotEndpoint::CloseTransportNoexcept() noexcept {
    cmd_subscriber_.reset();
    state_publisher_.reset();
}
