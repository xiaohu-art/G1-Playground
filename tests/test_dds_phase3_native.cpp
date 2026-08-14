#include <cassert>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "g1_dds_robot_endpoint.hpp"
#include "dds_utils.hpp"

using unitree_cpp_detail::DdsCommandValidation;

template <typename Message>
std::uint32_t MessageCrc(Message& message) {
    return unitree_cpp_detail::Crc32Core(
        reinterpret_cast<const std::uint32_t*>(&message),
        static_cast<std::uint32_t>((sizeof(Message) >> 2) - 1));
}

int main() {
    const std::array<std::uint32_t, 4> crc_fixture = {
        0x00000000U, 0x12345678U, 0x9ABCDEF0U, 0xFFFFFFFFU};
    assert(unitree_cpp_detail::Crc32Core(crc_fixture.data(), crc_fixture.size()) == 0x85C09DADU);

    unitree_hg::msg::dds_::LowCmd_ command;
    command.mode_pr(0);
    command.mode_machine(5);
    for (std::size_t index = 0; index < 29; ++index) {
        auto& motor = command.motor_cmd().at(index);
        motor.mode(1);
        motor.q(static_cast<float>(index));
        motor.dq(static_cast<float>(index) + 0.25F);
        motor.tau(static_cast<float>(index) + 0.5F);
        motor.kp(static_cast<float>(index) + 1.0F);
        motor.kd(static_cast<float>(index) + 2.0F);
    }
    command.crc(MessageCrc(command));

    DdsCommandSnapshot snapshot;
    assert(unitree_cpp_detail::ValidateDdsCommandForTest(command, 5, &snapshot) ==
           DdsCommandValidation::ACCEPTED);
    assert(snapshot.valid);
    assert(snapshot.q.size() == 29);
    assert(snapshot.q.front() == 0.0);
    assert(snapshot.kd.back() == 30.0);

    auto bad_crc = command;
    bad_crc.crc(bad_crc.crc() + 1);
    assert(unitree_cpp_detail::ValidateDdsCommandForTest(bad_crc, 5, &snapshot) ==
           DdsCommandValidation::CRC_ERROR);

    auto bad_mode = command;
    bad_mode.mode_machine(4);
    bad_mode.crc(MessageCrc(bad_mode));
    assert(unitree_cpp_detail::ValidateDdsCommandForTest(bad_mode, 5, &snapshot) ==
           DdsCommandValidation::MODE_ERROR);

    auto bad_value = command;
    bad_value.motor_cmd().at(7).q(std::numeric_limits<float>::quiet_NaN());
    bad_value.crc(MessageCrc(bad_value));
    assert(unitree_cpp_detail::ValidateDdsCommandForTest(bad_value, 5, &snapshot) ==
           DdsCommandValidation::VALUE_ERROR);

    DdsLowStateSnapshot state;
    for (std::size_t index = 0; index < 29; ++index) {
        state.q[index] = static_cast<double>(index) * 0.1;
        state.dq[index] = static_cast<double>(index) * 0.2;
        state.tau_est[index] = static_cast<double>(index) * 0.3;
    }
    state.quaternion = {1.0, 0.1, 0.2, 0.3};
    state.gyroscope = {0.4, 0.5, 0.6};
    state.wireless_remote[3] = 9;
    unitree_hg::msg::dds_::LowState_ low_state;
    const auto crc = unitree_cpp_detail::FillDdsLowStateForTest(state, 5, 1, &low_state);
    assert(low_state.tick() == 1);
    assert(low_state.mode_pr() == 0);
    assert(low_state.mode_machine() == 5);
    assert(low_state.motor_state().at(28).q() == static_cast<float>(2.8));
    assert(low_state.motor_state().at(29).q() == 0.0F);
    assert(low_state.imu_state().quaternion().at(0) == 1.0F);
    assert(low_state.wireless_remote().at(3) == 9);
    assert(crc == low_state.crc());
    assert(crc == MessageCrc(low_state));
    return 0;
}
