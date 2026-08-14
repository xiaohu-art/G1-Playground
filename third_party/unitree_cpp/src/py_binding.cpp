#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "dds_sim_server.hpp"
#include "unitree_controller.hpp"

namespace py = pybind11;

void bind_DdsSimServer(py::module_& m) {
    py::class_<DdsCommandSnapshot>(m, "DdsCommandSnapshot")
        .def(py::init<>())
        .def_readonly("valid", &DdsCommandSnapshot::valid)
        .def_readonly("sequence", &DdsCommandSnapshot::sequence)
        .def_readonly("age_seconds", &DdsCommandSnapshot::age_seconds)
        .def_readonly("mode_pr", &DdsCommandSnapshot::mode_pr)
        .def_readonly("mode_machine", &DdsCommandSnapshot::mode_machine)
        .def_readonly("q", &DdsCommandSnapshot::q)
        .def_readonly("dq", &DdsCommandSnapshot::dq)
        .def_readonly("tau", &DdsCommandSnapshot::tau)
        .def_readonly("kp", &DdsCommandSnapshot::kp)
        .def_readonly("kd", &DdsCommandSnapshot::kd);

    py::class_<DdsLowStateSnapshot>(m, "DdsLowStateSnapshot")
        .def(py::init<>())
        .def_readwrite("q", &DdsLowStateSnapshot::q)
        .def_readwrite("dq", &DdsLowStateSnapshot::dq)
        .def_readwrite("tau_est", &DdsLowStateSnapshot::tau_est)
        .def_readwrite("quaternion", &DdsLowStateSnapshot::quaternion)
        .def_readwrite("gyroscope", &DdsLowStateSnapshot::gyroscope)
        .def_readwrite("accelerometer", &DdsLowStateSnapshot::accelerometer)
        .def_readwrite("rpy", &DdsLowStateSnapshot::rpy)
        .def_property(
            "wireless_remote",
            [](const DdsLowStateSnapshot& self) {
                return py::bytes(reinterpret_cast<const char*>(self.wireless_remote.data()), self.wireless_remote.size());
            },
            [](DdsLowStateSnapshot& self, py::bytes value) {
                std::string bytes = value;
                if (bytes.size() != self.wireless_remote.size()) {
                    throw std::invalid_argument("wireless_remote must contain 40 bytes");
                }
                std::copy(bytes.begin(), bytes.end(), self.wireless_remote.begin());
            });

    py::class_<DdsSimServerStats>(m, "DdsSimServerStats")
        .def_readonly("accepted_commands", &DdsSimServerStats::accepted_commands)
        .def_readonly("crc_errors", &DdsSimServerStats::crc_errors)
        .def_readonly("finite_errors", &DdsSimServerStats::finite_errors)
        .def_readonly("mode_errors", &DdsSimServerStats::mode_errors);

    py::class_<G1DdsSimServer>(m, "G1DdsSimServer")
        .def(py::init([](py::dict cfg_dict) {
            DdsSimServerConfig cfg;
            cfg.domain_id = cfg_dict["domain_id"].cast<std::int32_t>();
            cfg.net_if = cfg_dict["net_if"].cast<std::string>();
            cfg.lowcmd_topic = cfg_dict["lowcmd_topic"].cast<std::string>();
            cfg.lowstate_topic = cfg_dict["lowstate_topic"].cast<std::string>();
            cfg.mode_machine = cfg_dict["mode_machine"].cast<std::uint8_t>();
            return new G1DdsSimServer(cfg);
        }))
        .def("get_command", &G1DdsSimServer::get_command)
        .def("publish_lowstate", &G1DdsSimServer::publish_lowstate)
        .def_property_readonly("stats", &G1DdsSimServer::stats)
        .def("close", &G1DdsSimServer::close);
}

void bind_ControllerState(py::module_& m) {
    py::enum_<unitree_cpp_detail::ControllerState>(m, "ControllerState")
        .value("RECEIVING", unitree_cpp_detail::ControllerState::RECEIVING)
        .value("ACTIVE", unitree_cpp_detail::ControllerState::ACTIVE)
        .value("CLOSED", unitree_cpp_detail::ControllerState::CLOSED)
        .export_values();
}

void bind_UnitreeConfig(py::module_& m) {
    py::class_<UnitreeConfig>(m, "UnitreeConfig")
        .def(py::init<>())
        .def_readwrite("net_if", &UnitreeConfig::net_if)
        .def_readwrite("domain_id", &UnitreeConfig::domain_id)
        .def_readwrite("control_dt", &UnitreeConfig::control_dt)
        .def_readwrite("msg_type", &UnitreeConfig::msg_type)
        .def_readwrite("control_mode", &UnitreeConfig::control_mode)
        .def_readwrite("hand_type", &UnitreeConfig::hand_type)
        .def_readwrite("lowcmd_topic", &UnitreeConfig::lowcmd_topic)
        .def_readwrite("lowstate_topic", &UnitreeConfig::lowstate_topic)
        .def_readwrite("enable_odometry", &UnitreeConfig::enable_odometry)
        .def_readwrite("sport_state_topic", &UnitreeConfig::sport_state_topic)
        .def_readwrite("stiffness", &UnitreeConfig::stiffness)
        .def_readwrite("damping", &UnitreeConfig::damping)
        .def_readwrite("num_dofs", &UnitreeConfig::num_dofs)
        .def_readwrite("motion_switcher_required", &UnitreeConfig::motion_switcher_required);
}

void bind_RobotState(py::module_& m) {
    py::class_<MotorState>(m, "MotorState")
        .def(py::init<size_t>())
        .def_readwrite("q", &MotorState::q)
        .def_readwrite("dq", &MotorState::dq)
        .def_readwrite("tau_est", &MotorState::tau_est);

    py::class_<ImuState>(m, "ImuState")
        .def(py::init<>())
        .def_readwrite("rpy", &ImuState::rpy)
        .def_readwrite("gyroscope", &ImuState::gyroscope)
        .def_readwrite("quaternion", &ImuState::quaternion)
        .def_readwrite("accelerometer", &ImuState::accelerometer);

    py::class_<RobotState>(m, "RobotState")
        .def(py::init<size_t>())
        .def_readwrite("tick", &RobotState::tick)
        .def_readwrite("mode_machine", &RobotState::mode_machine)
        .def_readwrite("motor_state", &RobotState::motor_state)
        .def_readwrite("imu_state", &RobotState::imu_state)
        // .def_readwrite("wireless_remote", &RobotState::wireless_remote);
        .def_property(
            "wireless_remote",
            [](const RobotState& self) {
                return py::bytes(reinterpret_cast<const char*>(self.wireless_remote), 40);
            },
            [](RobotState& self, py::bytes b) {
                std::string buf = b;
                if (buf.size() != 40) {
                    throw std::runtime_error("Expected 40 bytes for wireless_remote");
                }
                std::memcpy(self.wireless_remote, buf.data(), 40);
            });

    py::class_<SportState>(m, "SportState")
        .def(py::init<>())
        .def_readwrite("position", &SportState::position)
        .def_readwrite("velocity", &SportState::velocity);
}

// UnitreeController Class
void bind_UnitreeController(py::module_& m) {
    py::class_<UnitreeController>(m, "UnitreeController")
        .def(py::init([](py::dict cfg_dict) {
            UnitreeConfig cfg;

            cfg.net_if = cfg_dict["net_if"].cast<std::string>();
            cfg.domain_id = cfg_dict["domain_id"].cast<std::int32_t>();
            cfg.control_dt = cfg_dict["control_dt"].cast<double>();
            cfg.msg_type = cfg_dict["msg_type"].cast<std::string>();
            cfg.hand_type = cfg_dict["hand_type"].cast<std::string>();
            cfg.lowcmd_topic = cfg_dict["lowcmd_topic"].cast<std::string>();
            cfg.lowstate_topic = cfg_dict["lowstate_topic"].cast<std::string>();
            cfg.sport_state_topic = cfg_dict["sport_state_topic"].cast<std::string>();
            cfg.enable_odometry = cfg_dict["enable_odometry"].cast<bool>();
            cfg.stiffness = cfg_dict["stiffness"].cast<std::vector<double>>();
            cfg.damping = cfg_dict["damping"].cast<std::vector<double>>();
            cfg.num_dofs = cfg_dict["num_dofs"].cast<unsigned short>();
            if (cfg_dict.contains("motion_switcher_required")) {
                cfg.motion_switcher_required = cfg_dict["motion_switcher_required"].cast<bool>();
            }

            std::string mode_str = cfg_dict["control_mode"].cast<std::string>();
            if (mode_str == "position")
                cfg.control_mode = ControlMode::POSITION;
            else if (mode_str == "velocity")
                cfg.control_mode = ControlMode::VELOCITY;
            else if (mode_str == "torque")
                cfg.control_mode = ControlMode::TORQUE;
            else
                throw std::invalid_argument("Invalid control_mode");

            return new UnitreeController(cfg);
        }))
        .def(py::init<const UnitreeConfig&>(), py::arg("config"))
        .def("activate_commands", &UnitreeController::activate_commands)
        .def_property_readonly("lifecycle_state", &UnitreeController::lifecycle_state)
        .def("self_check", &UnitreeController::self_check)
        .def("step", &UnitreeController::step, py::arg("actions"))
        .def("step_hands", &UnitreeController::step_hands, py::arg("l_hand_pose"), py::arg("r_hand_pose"))
        .def("set_gains", &UnitreeController::set_gains, py::arg("stiffness"), py::arg("damping"))
        .def("shutdown", &UnitreeController::shutdown)
        .def("get_robot_state", &UnitreeController::get_robot_state)
        .def("get_sport_state", &UnitreeController::get_sport_state);
}

PYBIND11_MODULE(unitree_cpp, m) {
    m.doc() = "pybind11 bindings for UnitreeController";

    bind_DdsSimServer(m);
    bind_ControllerState(m);
    // bind_ControlMode(m);
    bind_UnitreeConfig(m);
    bind_RobotState(m);
    bind_UnitreeController(m);
}
