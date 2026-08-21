// src/g1_dds_control_endpoint.hpp
#pragma once
#include "dds_utils.hpp"

#include <atomic>
#include <vector>
#include <map>
#include <string>

#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>

#include <unitree/idl/go2/SportModeState_.hpp>

using std::size_t;
using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace unitree_go::msg::dds_;

template <typename T>
class DataBuffer {
   public:
    void SetData(const T& newData) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        data = std::make_shared<T>(newData);
    }

    std::shared_ptr<const T> GetData() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return data ? data : nullptr;
    }

    void Clear() {
        std::unique_lock<std::shared_mutex> lock(mutex);
        data = nullptr;
    }

   private:
    std::shared_ptr<T> data;
    mutable std::shared_mutex mutex;
};

struct MotorCommand {
    std::vector<float> q_target;
    std::vector<float> dq_target;
    std::vector<float> kp;
    std::vector<float> kd;
    std::vector<float> tau_ff;
    MotorCommand(size_t num_motors) : q_target(num_motors, 0.0f),
                                      dq_target(num_motors, 0.0f),
                                      kp(num_motors, 0.0f),
                                      kd(num_motors, 0.0f),
                                      tau_ff(num_motors, 0.0f) {}
};

struct HandCommand {
    std::vector<float> q_target;
    std::vector<float> dq_target;
    std::vector<float> kp;
    std::vector<float> kd;
    std::vector<float> tau_ff;
    HandCommand(size_t num_motors) : q_target(num_motors, 0.0f),
                                     dq_target(num_motors, 0.0f),
                                     kp(num_motors, 0.0f),
                                     kd(num_motors, 0.0f),
                                     tau_ff(num_motors, 0.0f) {}
};

struct MotorState {
    std::vector<float> q;
    std::vector<float> dq;
    std::vector<float> tau_est;

    MotorState(size_t num_motors) : q(num_motors, 0.0f),
                                    dq(num_motors, 0.0f),
                                    tau_est(num_motors, 0.0f) {}
};

struct ImuState {
    std::array<float, 3> rpy;
    std::array<float, 3> gyroscope;
    std::array<float, 4> quaternion;
    std::array<float, 3> accelerometer;
    ImuState() : rpy({0.0f, 0.0f, 0.0f}),
                 gyroscope({0.0f, 0.0f, 0.0f}),
                 quaternion({1.0f, 0.0f, 0.0f, 0.0f}),  // Default to no rotation
                 accelerometer({0.0f, 0.0f, 0.0f})
    {}
};

struct RobotState {
    uint32_t tick;
    uint8_t mode_machine;
    MotorState motor_state;
    ImuState imu_state;
    uint8_t wireless_remote[40] = {};
    std::chrono::steady_clock::time_point received_at;

    RobotState(size_t num_motors) : tick(0),
                                    mode_machine(0),
                                    motor_state(num_motors) {}
};

struct SportState {
    std::array<float, 3> position{};
    std::array<float, 3> velocity{};
    float body_height = 0.0F;
    std::chrono::steady_clock::time_point received_at;
};

enum class Mode {
    PR = 0,  // Series Control for Ptich/Roll Joints
    AB = 1   // Parallel Control for A/B Joints
};

enum class ControlMode {
    POSITION = 0,
    VELOCITY = 1,
    TORQUE = 2
};

namespace unitree_cpp_detail {

enum class DdsControlEndpointState {
    RECEIVING = 0,
    ACTIVE = 1,
    CLOSED = 2,
};

bool IsValidLowStateCandidate(const RobotState& candidate, const RobotState* previous, std::uint8_t expected_mode);
bool IsFreshTimestamp(
    std::chrono::steady_clock::time_point received_at,
    std::chrono::steady_clock::time_point now,
    double timeout_seconds);

// Small, DDS-free lifecycle core. The callbacks make the actual side effects
// injectable in unit tests while keeping one source of truth for the control endpoint
// state in production.
class DdsControlEndpointLifecycle {
   public:
    DdsControlEndpointState state() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

    template <class Activator>
    bool ActivateOnce(Activator&& activator) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == DdsControlEndpointState::ACTIVE) {
            return false;
        }
        if (state_ == DdsControlEndpointState::CLOSED) {
            throw std::logic_error("cannot activate a closed G1DdsControlEndpoint");
        }

        // Keep RECEIVING if activation throws, so callers may fix the cause and
        // retry without reconstructing the state subscriber.
        std::forward<Activator>(activator)();
        state_ = DdsControlEndpointState::ACTIVE;
        return true;
    }

    template <class Operation>
    void RunWhileActive(Operation&& operation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != DdsControlEndpointState::ACTIVE) {
            throw std::logic_error("G1DdsControlEndpoint command transport is not active");
        }
        std::forward<Operation>(operation)();
    }

    template <class Operation>
    void RunWhileOpen(Operation&& operation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == DdsControlEndpointState::CLOSED) {
            throw std::logic_error("G1DdsControlEndpoint is closed");
        }
        std::forward<Operation>(operation)();
    }

    template <class Closer>
    bool CloseOnce(Closer&& closer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == DdsControlEndpointState::CLOSED) {
            return false;
        }

        const DdsControlEndpointState previous_state = state_;
        state_ = DdsControlEndpointState::CLOSED;
        std::forward<Closer>(closer)(previous_state);
        return true;
    }

   private:
    mutable std::mutex mutex_;
    DdsControlEndpointState state_ = DdsControlEndpointState::RECEIVING;
};

}  // namespace unitree_cpp_detail

struct G1DdsControlEndpointConfig {
    std::string net_if;
    std::int32_t domain_id = 0;
    double control_dt;

    std::string msg_type;      // "hg" or "go"
    ControlMode control_mode;  // "position", etc.
    std::string hand_type;     // "Dex-3" or "NONE"

    std::string lowcmd_topic;
    std::string lowstate_topic;

    bool enable_odometry;
    std::string sport_state_topic;

    std::vector<double> stiffness;
    std::vector<double> damping;
    unsigned short num_dofs;
    bool motion_switcher_required = true;
};

class G1DdsControlEndpoint {
   public:
    G1DdsControlEndpoint(const G1DdsControlEndpointConfig& cfg);
    ~G1DdsControlEndpoint();
    bool activate_commands();
    unitree_cpp_detail::DdsControlEndpointState lifecycle_state() const;
    bool self_check();
    void step(const std::vector<double>& actions);
    void step_hands(const std::vector<double>& l_hand_pose, const std::vector<double>& r_hand_pose);
    void set_gains(const std::vector<double>& stiffness, const std::vector<double>& damping);
    void shutdown();

    RobotState get_robot_state();
    SportState get_sport_state();

   private:
    G1DdsControlEndpointConfig cfg_;

    std::vector<double> stiffness_;
    std::vector<double> damping_;
    unsigned short num_dofs_;
    unsigned short num_dofs_hand_;

    Mode mode_pr_;
    unitree_cpp_detail::DdsControlEndpointLifecycle lifecycle_;
    std::atomic<std::int64_t> last_command_time_ns_{0};
    std::atomic<bool> command_watchdog_fired_{false};
    std::mutex command_write_mutex_;
    std::mutex state_callback_mutex_;

    DataBuffer<MotorCommand> motor_command_buffer_;

    // DataBuffer<MotorState> motor_state_buffer_;
    // DataBuffer<ImuState> imu_state_buffer_;
    // DataBuffer<std::array<uint8_t, 40>> wireless_remote_buffer_;

    // DataBuffer<LowState_> low_state_buffer_;
    DataBuffer<RobotState> robot_state_buffer_;
    DataBuffer<SportState> sport_state_buffer_;

    ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
    // ChannelSubscriberPtr<IMUState_> imutorso_subscriber_;
    ThreadPtr command_writer_ptr_;

    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> estimate_state_subscriber;

    DataBuffer<HandCommand> hand_command_left_buffer_;
    DataBuffer<HandCommand> hand_command_right_buffer_;
    ChannelPublisherPtr<HandCmd_> handcmd_left_publisher_;
    ChannelPublisherPtr<HandCmd_> handcmd_right_publisher_;
    ThreadPtr handcmd_writer_ptr_;

    std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;

    void InitializeObserver();
    void InitializeCommandTransport();
    void StopCommandWritersNoexcept() noexcept;
    void CloseCommandTransportNoexcept() noexcept;
    void CloseObserverTransportNoexcept() noexcept;
    void CloseTransportNoexcept() noexcept;
    void SendDampingCommand();
    void ValidateRobotState(const RobotState& state) const;
    bool HasFreshRobotState() const;
    bool CommandExpired() const;
    void WriteMotorCommand(const MotorCommand& command);
    void LowCommandWriterLocked();
    void LowStateHandler(const void* message);
    void SportStateHandler(const void* message);
    void LowCommandWriter();
    void HandCommandWriter();
};
