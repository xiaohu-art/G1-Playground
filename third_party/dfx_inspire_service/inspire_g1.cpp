#include "inspire.h"
#include "param.h"
#include "hand_worker.h"

#include "dds/Publisher.h"
#include "dds/Subscription.h"
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/common/thread/recurrent_thread.hpp>

#include <chrono>

class InspireRunner
{
public:
  InspireRunner()
  {
    serial1 = std::make_shared<SerialPort>("/dev/ttyUSB0", B115200);
    serial2 = std::make_shared<SerialPort>("/dev/ttyUSB1", B115200);

    righthand = std::make_shared<HandWorker>(
      std::make_shared<inspire::InspireHand>(serial2, 1), "right");
    lefthand = std::make_shared<HandWorker>(
      std::make_shared<inspire::InspireHand>(serial1, 1), "left");

    handcmd = std::make_shared<unitree::robot::SubscriptionBase<unitree_go::msg::dds_::MotorCmds_>>(
        "rt/" + param::ns + "/cmd");
    handcmd->msg_.cmds().resize(12);
    handstate = std::make_unique<unitree::robot::RealTimePublisher<unitree_go::msg::dds_::MotorStates_>>(
        "rt/" + param::ns + "/state");
    handstate->msg_.states().resize(12);

    righthand->start();
    lefthand->start();

    report_time = std::chrono::steady_clock::now();
    report_right = 0;
    report_left = 0;

    thread = std::make_shared<unitree::common::RecurrentThread>(
      10000, std::bind(&InspireRunner::run, this)
    );
  }

  ~InspireRunner()
  {
    righthand->stop();
    lefthand->stop();
  }

  void run()
  {
    const bool active = !handcmd->isTimeout();
    righthand->set_command_active(active);
    lefthand->set_command_active(active);

    if (active)
    {
      Eigen::Matrix<double, 12, 1> qcmd;
      for (int i(0); i < 12; i++)
      {
        qcmd(i) = handcmd->msg_.cmds()[i].q();
      }
      righthand->set_target(qcmd.block<6, 1>(0, 0));
      lefthand->set_target(qcmd.block<6, 1>(6, 0));
    }

    HandWorker::Vector6 qr, ql, dqr, dql;
    uint32_t lost_r(0), lost_l(0);
    const bool ok_r = righthand->get_state(qr, dqr, lost_r);
    const bool ok_l = lefthand->get_state(ql, dql, lost_l);

    if (handstate->trylock())
    {
      for (int i(0); i < 6; i++)
      {
        handstate->msg_.states()[i].q() = ok_r ? qr(i) : 0.0;
        handstate->msg_.states()[i].dq() = ok_r ? dqr(i) : 0.0;
        handstate->msg_.states()[i].lost() = lost_r;
        handstate->msg_.states()[i + 6].q() = ok_l ? ql(i) : 0.0;
        handstate->msg_.states()[i + 6].dq() = ok_l ? dql(i) : 0.0;
        handstate->msg_.states()[i + 6].lost() = lost_l;
      }
      handstate->unlockAndPublish();
    }

    report();
  }

  void report()
  {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - report_time).count();
    if (elapsed < 2.0) return;

    const uint64_t cr = righthand->cycles();
    const uint64_t cl = lefthand->cycles();
    spdlog::info(
      "serial cycles/s  right={:.1f}  left={:.1f}  (cmd {})",
      (cr - report_right) / elapsed,
      (cl - report_left) / elapsed,
      handcmd->isTimeout() ? "idle" : "active");
    report_time = now;
    report_right = cr;
    report_left = cl;
  }

  unitree::common::ThreadPtr thread;

  SerialPort::SharedPtr serial1;
  SerialPort::SharedPtr serial2;
  HandWorker::SharedPtr lefthand;
  HandWorker::SharedPtr righthand;

  std::chrono::steady_clock::time_point report_time;
  uint64_t report_right;
  uint64_t report_left;

  std::unique_ptr<unitree::robot::RealTimePublisher<unitree_go::msg::dds_::MotorStates_>> handstate;
  std::shared_ptr<unitree::robot::SubscriptionBase<unitree_go::msg::dds_::MotorCmds_>> handcmd;
};

int main(int argc, char ** argv)
{
  auto vm = param::helper(argc, argv);
  unitree::robot::ChannelFactory::Instance()->Init(0, param::network);

  std::cout << " --- Unitree Robotics --- " << std::endl;
  std::cout << "  Inspire Hand Controller  " << std::endl;
  std::cout << "  (parallel serial workers) " << std::endl;

  InspireRunner runner;

  while (true)
  {
    sleep(1);
  }
  return 0;
}
