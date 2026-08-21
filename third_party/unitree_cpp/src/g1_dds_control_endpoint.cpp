#include "g1_dds_control_endpoint.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using std::size_t;
using SteadyClock = std::chrono::steady_clock;

namespace {

bool InitChannelFactoryOnce(const G1DdsControlEndpointConfig& cfg) {
    return unitree_cpp_detail::InitializeDdsEndpointOnce(cfg.domain_id, cfg.net_if);
}

std::int64_t NowNanoseconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now().time_since_epoch()).count();
}

double ControlEndpointTimeout(const G1DdsControlEndpointConfig& cfg) {
    return std::max(5.0 * cfg.control_dt, 0.1);
}

bool TickAdvanced(std::uint32_t previous, std::uint32_t current) {
    return current != previous && static_cast<std::int32_t>(current - previous) > 0;
}

template <typename Values>
bool AllFinite(const Values& values) {
    return std::all_of(values.begin(), values.end(), [](auto value) { return std::isfinite(value); });
}

template <typename Values>
bool AllRepresentableAsFloat(const Values& values) {
    constexpr double kMaxFloat = std::numeric_limits<float>::max();
    return std::all_of(values.begin(), values.end(), [kMaxFloat](auto value) {
        const double converted = static_cast<double>(value);
        return std::isfinite(converted) && std::abs(converted) <= kMaxFloat;
    });
}

}  // namespace

bool unitree_cpp_detail::IsValidLowStateCandidate(
    const RobotState& candidate, const RobotState* previous, std::uint8_t expected_mode) {
    if (candidate.mode_machine != expected_mode || !AllFinite(candidate.motor_state.q) ||
        !AllFinite(candidate.motor_state.dq) || !AllFinite(candidate.motor_state.tau_est) ||
        !AllFinite(candidate.imu_state.quaternion) || !AllFinite(candidate.imu_state.gyroscope) ||
        !AllFinite(candidate.imu_state.accelerometer) || !AllFinite(candidate.imu_state.rpy)) {
        return false;
    }
    return previous == nullptr || TickAdvanced(previous->tick, candidate.tick);
}

bool unitree_cpp_detail::IsFreshTimestamp(
    SteadyClock::time_point received_at, SteadyClock::time_point now, double timeout_seconds) {
    const double age_seconds = std::chrono::duration<double>(now - received_at).count();
    return std::isfinite(age_seconds) && age_seconds >= 0 && age_seconds <= timeout_seconds;
}

G1DdsControlEndpoint::G1DdsControlEndpoint(const G1DdsControlEndpointConfig& cfg)
    : cfg_(cfg),
      stiffness_(cfg.stiffness),
      damping_(cfg.damping),
      num_dofs_(cfg.num_dofs),
      mode_pr_(Mode::PR) {
    if (!std::isfinite(cfg_.control_dt) || cfg_.control_dt <= 0 || num_dofs_ != 29 ||
        stiffness_.size() != num_dofs_ || damping_.size() != num_dofs_ || !AllRepresentableAsFloat(stiffness_) ||
        !AllRepresentableAsFloat(damping_)) {
        throw std::invalid_argument("G1DdsControlEndpoint requires finite 29-DoF gains and a positive control_dt");
    }
    if (cfg.hand_type == "Dex-3") {
        num_dofs_hand_ = 7;
    } else if (cfg.hand_type == "NONE") {
        num_dofs_hand_ = 0;
    } else {
        throw std::runtime_error("Unsupported hand type: " + cfg.hand_type);
    }

    const bool initialized_endpoint = InitChannelFactoryOnce(cfg_);
    std::cout << cfg.hand_type << " hand with " << num_dofs_hand_ << " DOFs." << std::endl;
    std::cout << "G1DdsControlEndpoint " << (initialized_endpoint ? "initialized" : "reused") << " DDS domain " << cfg_.domain_id
              << " on network interface: " << cfg_.net_if << std::endl;

    try {
        InitializeObserver();
    } catch (...) {
        CloseTransportNoexcept();
        throw;
    }
}

G1DdsControlEndpoint::~G1DdsControlEndpoint() {
    try {
        lifecycle_.CloseOnce([this](unitree_cpp_detail::DdsControlEndpointState) { CloseTransportNoexcept(); });
    } catch (...) {
        CloseTransportNoexcept();
    }
}

void G1DdsControlEndpoint::InitializeObserver() {
    lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(cfg_.lowstate_topic));
    lowstate_subscriber_->InitChannel(std::bind(&G1DdsControlEndpoint::LowStateHandler, this, std::placeholders::_1), 1);

    if (cfg_.enable_odometry) {
        std::cout << "Odometry enabled, subscribing to sport state topic: " << cfg_.sport_state_topic << std::endl;
        estimate_state_subscriber.reset(new ChannelSubscriber<SportModeState_>(cfg_.sport_state_topic));
        estimate_state_subscriber->InitChannel(std::bind(&G1DdsControlEndpoint::SportStateHandler, this, std::placeholders::_1), 1);
    } else {
        std::cout << "Odometry disabled." << std::endl;
    }
}

void G1DdsControlEndpoint::InitializeCommandTransport() {
    try {
        if (cfg_.motion_switcher_required) {
            msc_ = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
            msc_->Init();
            const auto release_deadline = SteadyClock::now() + std::chrono::seconds(10);
            std::string form, name;
            while (true) {
                const auto remaining = release_deadline - SteadyClock::now();
                if (remaining <= SteadyClock::duration::zero()) {
                    throw std::runtime_error("Timed out releasing the active motion control service");
                }
                msc_->SetTimeout(std::min(5.0F, std::chrono::duration<float>(remaining).count()));
                if (msc_->CheckMode(form, name) != 0) {
                    throw std::runtime_error("Motion control service is unavailable");
                }
                if (name.empty()) {
                    std::cout << "Motion control service is inactive." << std::endl;
                    break;
                }
                const auto release_remaining = release_deadline - SteadyClock::now();
                if (release_remaining <= SteadyClock::duration::zero()) {
                    throw std::runtime_error("Timed out releasing the active motion control service");
                }
                msc_->SetTimeout(std::min(5.0F, std::chrono::duration<float>(release_remaining).count()));
                if (msc_->ReleaseMode() != 0) {
                    throw std::runtime_error("Failed to release the active motion control service");
                }
                const auto sleep_remaining = release_deadline - SteadyClock::now();
                if (sleep_remaining <= SteadyClock::duration::zero()) {
                    throw std::runtime_error("Timed out releasing the active motion control service");
                }
                std::this_thread::sleep_for(
                    std::min(std::chrono::duration_cast<SteadyClock::duration>(std::chrono::seconds(1)), sleep_remaining));
            }
        } else {
            std::cout << "Motion control service is disabled for this DDS endpoint." << std::endl;
        }

        lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(cfg_.lowcmd_topic));  // TODO: switch Cmd Type
        lowcmd_publisher_->InitChannel();
        command_writer_ptr_ = CreateRecurrentThreadEx(
            "command_writer", UT_CPU_ID_NONE, uint(cfg_.control_dt * 1e6), &G1DdsControlEndpoint::LowCommandWriter, this);

        if (num_dofs_hand_ > 0) {
            handcmd_left_publisher_.reset(new ChannelPublisher<HandCmd_>("rt/dex3/left/cmd"));
            handcmd_left_publisher_->InitChannel();
            handcmd_right_publisher_.reset(new ChannelPublisher<HandCmd_>("rt/dex3/right/cmd"));
            handcmd_right_publisher_->InitChannel();
            handcmd_writer_ptr_ = CreateRecurrentThreadEx(
                "handcmd_writer", UT_CPU_ID_NONE, uint(cfg_.control_dt * 1e6 * 5), &G1DdsControlEndpoint::HandCommandWriter, this);
        }
    } catch (...) {
        CloseCommandTransportNoexcept();
        throw;
    }
}

bool G1DdsControlEndpoint::activate_commands() {
    return lifecycle_.ActivateOnce([this]() { InitializeCommandTransport(); });
}

unitree_cpp_detail::DdsControlEndpointState G1DdsControlEndpoint::lifecycle_state() const {
    return lifecycle_.state();
}

bool G1DdsControlEndpoint::self_check() {
    if (lifecycle_.state() == unitree_cpp_detail::DdsControlEndpointState::CLOSED) {
        std::cerr << "G1DdsControlEndpoint is closed." << std::endl;
        return false;
    }
    try {
        RobotState robot_state = get_robot_state();
        if (robot_state.tick == 0) {
            std::cerr << "Robot state tick is zero, no data received." << std::endl;
            return false;
        }
        if (cfg_.enable_odometry) {
            SportState sport_state = get_sport_state();
            if (sport_state.position.empty() || sport_state.velocity.empty()) {
                std::cerr << "Sport state data is empty." << std::endl;
                return false;
            }
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "No data available: " << e.what() << std::endl;
        return false;
    }
    std::cout << "G1DdsControlEndpoint self-check passed." << std::endl;
    return true;
}

void G1DdsControlEndpoint::LowStateHandler(const void* message) {
    LowState_ low_state = *(const LowState_*)message;
    // std::cout << "LowState received: " << low_state.tick() << std::endl;
    if (low_state.crc() != unitree_cpp_detail::Crc32Core(
                               reinterpret_cast<const std::uint32_t*>(&low_state), (sizeof(LowState_) >> 2) - 1)) {
        return;
    }
    std::lock_guard<std::mutex> callback_lock(state_callback_mutex_);
    // low_state_buffer_.SetData(low_state);

    RobotState robot_state_tmp(num_dofs_);

    robot_state_tmp.tick = low_state.tick();
    robot_state_tmp.mode_machine = low_state.mode_machine();
    robot_state_tmp.received_at = SteadyClock::now();

    // get motor state
    // MotorState ms_tmp(num_dofs_);
    MotorState& ms_tmp = robot_state_tmp.motor_state;
    for (int i = 0; i < num_dofs_; ++i) {
        ms_tmp.q.at(i) = low_state.motor_state()[i].q();
        ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
        ms_tmp.tau_est.at(i) = low_state.motor_state()[i].tau_est();
        // if (low_state.motor_state()[i].motorstate() && i <= RightAnkleRoll)
        //     std::cout << "[ERROR] motor " << i << " with code " << low_state.motor_state()[i].motorstate() << "\n";
    }
    // motor_state_buffer_.SetData(ms_tmp);

    // get imu state
    // ImuState imu_tmp;
    ImuState& imu_tmp = robot_state_tmp.imu_state;
    imu_tmp.quaternion = low_state.imu_state().quaternion();
    imu_tmp.gyroscope = low_state.imu_state().gyroscope();
    imu_tmp.accelerometer = low_state.imu_state().accelerometer();
    imu_tmp.rpy = low_state.imu_state().rpy();
    // imu_state_buffer_.SetData(imu_tmp);

    memcpy(&robot_state_tmp.wireless_remote, &low_state.wireless_remote()[0], 40);
    // std::cout << "imu rpy: " << imu_tmp.rpy[0] << ", " << imu_tmp.rpy[1] << ", " << imu_tmp.rpy[2] << std::endl;

    const std::shared_ptr<const RobotState> previous_state = robot_state_buffer_.GetData();
    if (!unitree_cpp_detail::IsValidLowStateCandidate(robot_state_tmp, previous_state.get(), 5)) {
        return;
    }

    if (!previous_state) {
        std::cout << "G1 type: " << unsigned(robot_state_tmp.mode_machine) << std::endl;
    }
    robot_state_buffer_.SetData(robot_state_tmp);
}

void G1DdsControlEndpoint::SportStateHandler(const void* message) {
    SportModeState_ estimator_state = *(const SportModeState_*)message;

    SportState sport_state_tmp;
    sport_state_tmp.position = estimator_state.position();
    sport_state_tmp.velocity = estimator_state.velocity();
    sport_state_tmp.body_height = estimator_state.body_height();
    if (!AllFinite(sport_state_tmp.position) || !AllFinite(sport_state_tmp.velocity) ||
        !std::isfinite(sport_state_tmp.body_height) || sport_state_tmp.body_height <= 0.0F) {
        return;
    }
    sport_state_tmp.received_at = SteadyClock::now();
    sport_state_buffer_.SetData(sport_state_tmp);
}

void G1DdsControlEndpoint::StopCommandWritersNoexcept() noexcept {
    const auto stop_thread = [](ThreadPtr& thread) {
        if (!thread) {
            return;
        }
        try {
            thread->Wait();
        } catch (...) {
        }
        thread.reset();
    };

    stop_thread(handcmd_writer_ptr_);
    stop_thread(command_writer_ptr_);
}

void G1DdsControlEndpoint::CloseCommandTransportNoexcept() noexcept {
    StopCommandWritersNoexcept();

    const auto close_publisher = [](auto& publisher) {
        if (!publisher) {
            return;
        }
        try {
            publisher->CloseChannel();
        } catch (...) {
        }
        publisher.reset();
    };

    close_publisher(handcmd_right_publisher_);
    close_publisher(handcmd_left_publisher_);
    close_publisher(lowcmd_publisher_);
    hand_command_right_buffer_.Clear();
    hand_command_left_buffer_.Clear();
    motor_command_buffer_.Clear();
    msc_.reset();
}

void G1DdsControlEndpoint::CloseObserverTransportNoexcept() noexcept {
    const auto close_subscriber = [](auto& subscriber) {
        if (!subscriber) {
            return;
        }
        try {
            subscriber->CloseChannel();
        } catch (...) {
        }
        subscriber.reset();
    };

    close_subscriber(estimate_state_subscriber);
    close_subscriber(lowstate_subscriber_);
}

void G1DdsControlEndpoint::CloseTransportNoexcept() noexcept {
    CloseCommandTransportNoexcept();
    CloseObserverTransportNoexcept();
}

void G1DdsControlEndpoint::LowCommandWriter() {
    std::lock_guard<std::mutex> write_lock(command_write_mutex_);
    LowCommandWriterLocked();
}

void G1DdsControlEndpoint::LowCommandWriterLocked() {
    const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
    if (!mc) {
        return;
    }
    if (CommandExpired() || !HasFreshRobotState()) {
        if (!command_watchdog_fired_.exchange(true)) {
            motor_command_buffer_.Clear();
            SendDampingCommand();
        }
        return;
    }
    LowCmd_ dds_low_command{};
    dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
    const std::shared_ptr<const RobotState> robot_state = robot_state_buffer_.GetData();
    if (!robot_state) {
        return;
    }
    dds_low_command.mode_machine() = robot_state->mode_machine;

    if (lowcmd_publisher_) {
        // std::cout << "LowCommandWriter called with motor command data." << std::endl;
        for (size_t i = 0; i < num_dofs_; i++) {
            dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
            dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
            dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
            dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
            dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
            dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
        }

        dds_low_command.crc() = unitree_cpp_detail::Crc32Core(
            reinterpret_cast<const std::uint32_t*>(&dds_low_command), (sizeof(dds_low_command) >> 2) - 1);
        if (!lowcmd_publisher_->Write(dds_low_command)) {
            command_watchdog_fired_.store(true);
            motor_command_buffer_.Clear();
            throw std::runtime_error("Failed to publish DDS LowCmd");
        }
    }
}

void G1DdsControlEndpoint::HandCommandWriter() {
    HandCmd_ dds_hand_command;

    dds_hand_command.motor_cmd().resize(num_dofs_hand_);

    const std::shared_ptr<const HandCommand> hc_l = hand_command_left_buffer_.GetData();
    if (hc_l && handcmd_left_publisher_) {
        // std::cout << "LowCommandWriter called with motor command data." << std::endl;
        for (size_t i = 0; i < num_dofs_hand_; i++) {
            dds_hand_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
            dds_hand_command.motor_cmd().at(i).tau() = hc_l->tau_ff.at(i);
            dds_hand_command.motor_cmd().at(i).q() = hc_l->q_target.at(i);
            dds_hand_command.motor_cmd().at(i).dq() = hc_l->dq_target.at(i);
            dds_hand_command.motor_cmd().at(i).kp() = hc_l->kp.at(i);
            dds_hand_command.motor_cmd().at(i).kd() = hc_l->kd.at(i);
        }

        handcmd_left_publisher_->Write(dds_hand_command);
    }
    const std::shared_ptr<const HandCommand> hc_r = hand_command_right_buffer_.GetData();
    if (hc_r && handcmd_right_publisher_) {
        // std::cout << "LowCommandWriter called with motor command data." << std::endl;
        for (size_t i = 0; i < num_dofs_hand_; i++) {
            dds_hand_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
            dds_hand_command.motor_cmd().at(i).tau() = hc_r->tau_ff.at(i);
            dds_hand_command.motor_cmd().at(i).q() = hc_r->q_target.at(i);
            dds_hand_command.motor_cmd().at(i).dq() = hc_r->dq_target.at(i);
            dds_hand_command.motor_cmd().at(i).kp() = hc_r->kp.at(i);
            dds_hand_command.motor_cmd().at(i).kd() = hc_r->kd.at(i);
        }

        handcmd_right_publisher_->Write(dds_hand_command);
    }
}

void G1DdsControlEndpoint::step(const std::vector<double>& actions) {
    lifecycle_.RunWhileActive([this, &actions]() {
        if (actions.size() != num_dofs_) {
            throw std::runtime_error("actions size mismatch");
        }
        if (!AllRepresentableAsFloat(actions)) {
            throw std::invalid_argument("actions must contain values representable by the DDS float wire type");
        }
        if (command_watchdog_fired_.load()) {
            throw std::runtime_error("G1DdsControlEndpoint command watchdog expired");
        }

        MotorCommand motor_command_tmp(num_dofs_);
        for (size_t i = 0; i < num_dofs_; ++i) {
            motor_command_tmp.kp.at(i) = stiffness_[i];
            motor_command_tmp.kd.at(i) = damping_[i];
            switch (cfg_.control_mode) {
                case ControlMode::POSITION:
                    motor_command_tmp.q_target.at(i) = actions[i];
                    break;
                case ControlMode::VELOCITY:
                    motor_command_tmp.dq_target.at(i) = actions[i];
                    break;
                case ControlMode::TORQUE:
                    motor_command_tmp.tau_ff.at(i) = actions[i];
                    break;
                default:
                    throw std::runtime_error("Unknown control mode");
            }
        }
        WriteMotorCommand(motor_command_tmp);
    });
}

void G1DdsControlEndpoint::step_hands(const std::vector<double>& l_hand_pose, const std::vector<double>& r_hand_pose) {
    lifecycle_.RunWhileActive([this, &l_hand_pose, &r_hand_pose]() {
        if (num_dofs_hand_ == 0) {
            throw std::logic_error("hand command transport is disabled");
        }
        if (l_hand_pose.size() != num_dofs_hand_ || r_hand_pose.size() != num_dofs_hand_) {
            throw std::runtime_error("l_hand_pose or r_hand_pose size mismatch");
        }

        HandCommand hand_command_left_tmp(num_dofs_hand_);
        for (size_t i = 0; i < num_dofs_hand_; ++i) {
            hand_command_left_tmp.q_target.at(i) = l_hand_pose[i];
            hand_command_left_tmp.dq_target.at(i) = 0.0;
            hand_command_left_tmp.kp.at(i) = 1.5f;
            hand_command_left_tmp.kd.at(i) = 0.1f;
            hand_command_left_tmp.tau_ff.at(i) = 0.0f;
        }
        hand_command_left_buffer_.SetData(hand_command_left_tmp);

        HandCommand hand_command_right_tmp(num_dofs_hand_);
        for (size_t i = 0; i < num_dofs_hand_; ++i) {
            hand_command_right_tmp.q_target.at(i) = r_hand_pose[i];
            hand_command_right_tmp.dq_target.at(i) = 0.0;
            hand_command_right_tmp.kp.at(i) = 1.5f;
            hand_command_right_tmp.kd.at(i) = 0.1f;
            hand_command_right_tmp.tau_ff.at(i) = 0.0f;
        }
        hand_command_right_buffer_.SetData(hand_command_right_tmp);
        HandCommandWriter();  // immediately send command
    });
}

void G1DdsControlEndpoint::set_gains(const std::vector<double>& stiffness, const std::vector<double>& damping) {
    lifecycle_.RunWhileOpen([this, &stiffness, &damping]() {
        if (stiffness.size() != num_dofs_ || damping.size() != num_dofs_) {
            throw std::runtime_error("stiffness or damping size mismatch");
        }
        if (!AllRepresentableAsFloat(stiffness) || !AllRepresentableAsFloat(damping)) {
            throw std::invalid_argument("gains must contain values representable by the DDS float wire type");
        }
        stiffness_ = stiffness;
        damping_ = damping;

        std::cout << "Gains set: stiffness = [";
        for (const auto& s : stiffness_) {
            std::cout << s << " ";
        }
        std::cout << "], damping = [";
        for (const auto& d : damping_) {
            std::cout << d << " ";
        }
        std::cout << "]" << std::endl;
    });
}

void G1DdsControlEndpoint::SendDampingCommand() {
    LowCmd_ dds_low_command{};
    dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
    const std::shared_ptr<const RobotState> robot_state = robot_state_buffer_.GetData();
    if (!robot_state) {
        return;
    }
    dds_low_command.mode_machine() = robot_state->mode_machine;
    for (size_t i = 0; i < num_dofs_; ++i) {
        auto& motor = dds_low_command.motor_cmd().at(i);
        motor.mode() = 1;
        motor.kd() = 5.0F;
    }
    dds_low_command.crc() = unitree_cpp_detail::Crc32Core(
        reinterpret_cast<const std::uint32_t*>(&dds_low_command), (sizeof(dds_low_command) >> 2) - 1);
    if (lowcmd_publisher_) {
        if (!lowcmd_publisher_->Write(dds_low_command)) {
            throw std::runtime_error("Failed to publish DDS damping LowCmd");
        }
    }
}

void G1DdsControlEndpoint::WriteMotorCommand(const MotorCommand& command) {
    std::lock_guard<std::mutex> write_lock(command_write_mutex_);
    if (command_watchdog_fired_.load()) {
        throw std::runtime_error("G1DdsControlEndpoint command watchdog expired");
    }
    motor_command_buffer_.SetData(command);
    last_command_time_ns_.store(NowNanoseconds());
    LowCommandWriterLocked();
}

bool G1DdsControlEndpoint::CommandExpired() const {
    const auto last_command_time = last_command_time_ns_.load();
    if (last_command_time == 0) {
        return false;
    }
    const double age_seconds = static_cast<double>(NowNanoseconds() - last_command_time) / 1e9;
    return age_seconds > ControlEndpointTimeout(cfg_);
}

bool G1DdsControlEndpoint::HasFreshRobotState() const {
    const std::shared_ptr<const RobotState> state = robot_state_buffer_.GetData();
    return state && unitree_cpp_detail::IsValidLowStateCandidate(*state, nullptr, 5) &&
           unitree_cpp_detail::IsFreshTimestamp(state->received_at, SteadyClock::now(), ControlEndpointTimeout(cfg_));
}

void G1DdsControlEndpoint::ValidateRobotState(const RobotState& state) const {
    if (!unitree_cpp_detail::IsValidLowStateCandidate(state, nullptr, 5)) {
        throw std::runtime_error("Low state data is invalid");
    }
    if (!unitree_cpp_detail::IsFreshTimestamp(state.received_at, SteadyClock::now(), ControlEndpointTimeout(cfg_))) {
        throw std::runtime_error("Low state data is stale");
    }
}

void G1DdsControlEndpoint::shutdown() {
    lifecycle_.CloseOnce([this](unitree_cpp_detail::DdsControlEndpointState previous_state) {
        std::cout << "Shutting down G1DdsControlEndpoint..." << std::endl;
        if (previous_state == unitree_cpp_detail::DdsControlEndpointState::ACTIVE) {
            // Stop periodic writers before emitting the single legacy damping
            // command. Observer shutdown never enters this branch.
            StopCommandWritersNoexcept();
            try {
                SendDampingCommand();
            } catch (...) {
                CloseTransportNoexcept();
                throw;
            }
        }
        CloseTransportNoexcept();
    });
}

RobotState G1DdsControlEndpoint::get_robot_state() {
    const std::shared_ptr<const RobotState> robot_state = robot_state_buffer_.GetData();

    if (robot_state) {
        ValidateRobotState(*robot_state);
        return *robot_state;
    } else {
        throw std::runtime_error("Low state data is not available");
    }
}

SportState G1DdsControlEndpoint::get_sport_state() {
    const std::shared_ptr<const SportState> sport_state = sport_state_buffer_.GetData();

    if (!sport_state) {
        throw std::runtime_error("Sport state data is not available");
    }
    if (!unitree_cpp_detail::IsFreshTimestamp(
            sport_state->received_at, SteadyClock::now(), ControlEndpointTimeout(cfg_))) {
        throw std::runtime_error("Sport state data is stale");
    }
    return *sport_state;
}

int main(int argc, char const* argv[]) {
    // Example usage of G1DdsControlEndpoint
    G1DdsControlEndpointConfig config;
    config.net_if = "enp13s0";
    config.domain_id = 0;
    config.control_dt = 0.1;
    config.msg_type = "hg";
    config.control_mode = ControlMode::POSITION;
    config.hand_type = "Dex-3";
    config.lowcmd_topic = "rt/lowcmd";
    config.lowstate_topic = "rt/lowstate";
    config.enable_odometry = false;
    config.sport_state_topic = "rt/odommodestate";
    config.stiffness = {1.0, 1.0, 1.0};  // Example stiffness values
    config.damping = {0.1, 0.1, 0.1};    // Example damping values
    config.num_dofs = 3;                 // Example number of DOFs

    G1DdsControlEndpoint control_endpoint(config);

    while (true)
        sleep(10);
    return 0;
}
