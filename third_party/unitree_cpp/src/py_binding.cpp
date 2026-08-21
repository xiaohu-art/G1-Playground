#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "g1_dds_control_endpoint.hpp"
#include "g1_dds_robot_endpoint.hpp"
#include "inspire_dds_endpoint.hpp"

namespace py = pybind11;

void bind_G1DdsRobotEndpoint(py::module_& m) {
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

    py::class_<G1DdsRobotEndpointStats>(m, "G1DdsRobotEndpointStats")
        .def_readonly("accepted_commands", &G1DdsRobotEndpointStats::accepted_commands)
        .def_readonly("crc_errors", &G1DdsRobotEndpointStats::crc_errors)
        .def_readonly("finite_errors", &G1DdsRobotEndpointStats::finite_errors)
        .def_readonly("mode_errors", &G1DdsRobotEndpointStats::mode_errors);

    py::class_<G1DdsRobotEndpoint>(m, "G1DdsRobotEndpoint")
        .def(py::init([](py::dict cfg_dict) {
            G1DdsRobotEndpointConfig cfg;
            cfg.domain_id = cfg_dict["domain_id"].cast<std::int32_t>();
            cfg.net_if = cfg_dict["net_if"].cast<std::string>();
            cfg.lowcmd_topic = cfg_dict["lowcmd_topic"].cast<std::string>();
            cfg.lowstate_topic = cfg_dict["lowstate_topic"].cast<std::string>();
            cfg.mode_machine = cfg_dict["mode_machine"].cast<std::uint8_t>();
            return new G1DdsRobotEndpoint(cfg);
        }))
        .def("get_command", &G1DdsRobotEndpoint::get_command)
        .def("publish_lowstate", &G1DdsRobotEndpoint::publish_lowstate)
        .def_property_readonly("stats", &G1DdsRobotEndpoint::stats)
        .def("close", &G1DdsRobotEndpoint::close);
}

void bind_DdsControlEndpointState(py::module_& m) {
    py::enum_<unitree_cpp_detail::DdsControlEndpointState>(m, "DdsControlEndpointState")
        .value("RECEIVING", unitree_cpp_detail::DdsControlEndpointState::RECEIVING)
        .value("ACTIVE", unitree_cpp_detail::DdsControlEndpointState::ACTIVE)
        .value("CLOSED", unitree_cpp_detail::DdsControlEndpointState::CLOSED)
        .export_values();
}

void bind_G1DdsControlEndpointConfig(py::module_& m) {
    py::class_<G1DdsControlEndpointConfig>(m, "G1DdsControlEndpointConfig")
        .def(py::init<>())
        .def_readwrite("net_if", &G1DdsControlEndpointConfig::net_if)
        .def_readwrite("domain_id", &G1DdsControlEndpointConfig::domain_id)
        .def_readwrite("control_dt", &G1DdsControlEndpointConfig::control_dt)
        .def_readwrite("msg_type", &G1DdsControlEndpointConfig::msg_type)
        .def_readwrite("control_mode", &G1DdsControlEndpointConfig::control_mode)
        .def_readwrite("hand_type", &G1DdsControlEndpointConfig::hand_type)
        .def_readwrite("lowcmd_topic", &G1DdsControlEndpointConfig::lowcmd_topic)
        .def_readwrite("lowstate_topic", &G1DdsControlEndpointConfig::lowstate_topic)
        .def_readwrite("enable_odometry", &G1DdsControlEndpointConfig::enable_odometry)
        .def_readwrite("sport_state_topic", &G1DdsControlEndpointConfig::sport_state_topic)
        .def_readwrite("stiffness", &G1DdsControlEndpointConfig::stiffness)
        .def_readwrite("damping", &G1DdsControlEndpointConfig::damping)
        .def_readwrite("num_dofs", &G1DdsControlEndpointConfig::num_dofs)
        .def_readwrite("motion_switcher_required", &G1DdsControlEndpointConfig::motion_switcher_required);
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
        .def_readwrite("velocity", &SportState::velocity)
        .def_readwrite("body_height", &SportState::body_height);
}

// G1DdsControlEndpoint Class
void bind_G1DdsControlEndpoint(py::module_& m) {
    py::class_<G1DdsControlEndpoint>(m, "G1DdsControlEndpoint")
        .def(py::init([](py::dict cfg_dict) {
            G1DdsControlEndpointConfig cfg;

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

            return new G1DdsControlEndpoint(cfg);
        }))
        .def(py::init<const G1DdsControlEndpointConfig&>(), py::arg("config"))
        .def("activate_commands", &G1DdsControlEndpoint::activate_commands)
        .def_property_readonly("lifecycle_state", &G1DdsControlEndpoint::lifecycle_state)
        .def("self_check", &G1DdsControlEndpoint::self_check)
        .def("step", &G1DdsControlEndpoint::step, py::arg("actions"))
        .def("step_hands", &G1DdsControlEndpoint::step_hands, py::arg("l_hand_pose"), py::arg("r_hand_pose"))
        .def("set_gains", &G1DdsControlEndpoint::set_gains, py::arg("stiffness"), py::arg("damping"))
        .def("shutdown", &G1DdsControlEndpoint::shutdown)
        .def("get_robot_state", &G1DdsControlEndpoint::get_robot_state)
        .def("get_sport_state", &G1DdsControlEndpoint::get_sport_state);
}

void bind_InspireDdsEndpoint(py::module_& m) {
    py::class_<InspireDdsEndpointConfig>(m, "InspireDdsEndpointConfig")
        .def(py::init<>())
        .def_readwrite("domain_id", &InspireDdsEndpointConfig::domain_id)
        .def_readwrite("net_if", &InspireDdsEndpointConfig::net_if)
        .def_readwrite("cmd_topic", &InspireDdsEndpointConfig::cmd_topic)
        .def_readwrite("state_topic", &InspireDdsEndpointConfig::state_topic);

    py::class_<InspireStateSnapshot>(m, "InspireStateSnapshot")
        .def(py::init<>())
        .def_readonly("valid", &InspireStateSnapshot::valid)
        .def_readonly("sequence", &InspireStateSnapshot::sequence)
        .def_readonly("age_seconds", &InspireStateSnapshot::age_seconds)
        .def_readonly("q", &InspireStateSnapshot::q)
        .def_readonly("dq", &InspireStateSnapshot::dq);

    py::class_<InspireCommandSnapshot>(m, "InspireCommandSnapshot")
        .def(py::init<>())
        .def_readonly("valid", &InspireCommandSnapshot::valid)
        .def_readonly("sequence", &InspireCommandSnapshot::sequence)
        .def_readonly("age_seconds", &InspireCommandSnapshot::age_seconds)
        .def_readonly("q", &InspireCommandSnapshot::q);

    auto from_dict = [](py::dict cfg_dict) {
        InspireDdsEndpointConfig cfg;
        if (cfg_dict.contains("domain_id")) cfg.domain_id = cfg_dict["domain_id"].cast<std::int32_t>();
        if (cfg_dict.contains("net_if")) cfg.net_if = cfg_dict["net_if"].cast<std::string>();
        if (cfg_dict.contains("cmd_topic")) cfg.cmd_topic = cfg_dict["cmd_topic"].cast<std::string>();
        if (cfg_dict.contains("state_topic")) cfg.state_topic = cfg_dict["state_topic"].cast<std::string>();
        return cfg;
    };

    py::class_<InspireDdsControlEndpoint>(m, "InspireDdsControlEndpoint")
        .def(py::init([from_dict](py::dict d) {
            return std::make_unique<InspireDdsControlEndpoint>(from_dict(d));
        }))
        .def("get_state", &InspireDdsControlEndpoint::get_state)
        .def("self_check", &InspireDdsControlEndpoint::self_check)
        .def("step", &InspireDdsControlEndpoint::step, py::arg("stroke"))
        .def("close", &InspireDdsControlEndpoint::close);

    py::class_<InspireDdsRobotEndpoint>(m, "InspireDdsRobotEndpoint")
        .def(py::init([from_dict](py::dict d) {
            return std::make_unique<InspireDdsRobotEndpoint>(from_dict(d));
        }))
        .def("get_command", &InspireDdsRobotEndpoint::get_command)
        .def("publish_state", &InspireDdsRobotEndpoint::publish_state,
             py::arg("stroke"), py::arg("stroke_rate"))
        .def("close", &InspireDdsRobotEndpoint::close);
}

PYBIND11_MODULE(unitree_cpp, m) {
    m.doc() = "pybind11 bindings for G1DdsControlEndpoint";

    bind_G1DdsRobotEndpoint(m);
    bind_DdsControlEndpointState(m);
    // bind_ControlMode(m);
    bind_G1DdsControlEndpointConfig(m);
    bind_RobotState(m);
    bind_G1DdsControlEndpoint(m);
    bind_InspireDdsEndpoint(m);
}
