#ifndef SERIAL_PORT_H
#define SERIAL_PORT_H

#include <termios.h>
#include <sys/select.h>
#include <string>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/serial.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <queue>

class SerialPort
{
public:
  using SharedPtr = std::shared_ptr<SerialPort>;

  SerialPort(std::string port, speed_t baudrate, int timeout_ms = 20)
  {
    set_timeout(timeout_ms);
    Init(port, baudrate);
  }

  ~SerialPort()
  {
    close(fd_);
  }

  ssize_t send(const uint8_t* data, size_t len)
  {
    tcflush(fd_, TCIFLUSH);
    ssize_t ret = ::write(fd_, data, len);
    return ret;
  }

  ssize_t recv(uint8_t* data, size_t len)
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    size_t got = 0;

    while (got < len)
    {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) break;

      const auto remain =
        std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();

      timeval tv;
      tv.tv_sec = remain / 1000000;
      tv.tv_usec = remain % 1000000;

      fd_set rset;
      FD_ZERO(&rset);
      FD_SET(fd_, &rset);

      const int ready = select(fd_ + 1, &rset, NULL, NULL, &tv);
      if (ready < 0)
      {
        if (errno == EINTR) continue;
        break;
      }
      if (ready == 0) break;

      const ssize_t n = ::read(fd_, data + got, len - got);
      if (n <= 0) break;
      got += static_cast<size_t>(n);
    }

    return static_cast<ssize_t>(got);
  }

  void set_timeout(int timeout_ms)
  {
    timeout_ = std::chrono::milliseconds(timeout_ms);
  }

private:
  void Init(std::string port, speed_t baudrate)
  {
    int ret;
    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0)
    {
      printf("Open serial port %s failed\n", port.c_str());
      exit(-1);
    }

    struct termios option;
    memset(&option, 0, sizeof(option));
    ret = tcgetattr(fd_, &option);

    option.c_oflag = 0;
    option.c_lflag = 0;
    option.c_iflag = 0;

    cfsetispeed(&option, baudrate);
    cfsetospeed(&option, baudrate);

    option.c_cflag &= ~CSIZE;
    option.c_cflag |= CS8;
    option.c_cflag &= ~PARENB;
    option.c_iflag &= ~INPCK;
    option.c_cflag &= ~CSTOPB;

    option.c_cc[VTIME] = 0;
    option.c_cc[VMIN] = 0;
    option.c_lflag |= CBAUDEX;

    ret = tcflush(fd_, TCIFLUSH);
    ret = tcsetattr(fd_, TCSANOW, &option);
    (void)ret;
  }

  int fd_;
  std::chrono::steady_clock::duration timeout_;
};

#endif // SERIAL_PORT_H
