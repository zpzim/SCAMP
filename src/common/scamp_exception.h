#ifndef SCAMP_SCAMP_EXCEPTION_H
#define SCAMP_SCAMP_EXCEPTION_H

#include <exception>
#include <string>

class SCAMPException : public std::exception {
 public:
  SCAMPException(std::string message) : _msg(std::move(message)) {}

  virtual ~SCAMPException() {}

  SCAMPException(const SCAMPException& copyFrom) = default;
  SCAMPException& operator=(const SCAMPException& copyFrom) = default;
  SCAMPException(SCAMPException&&) = default;
  SCAMPException& operator=(SCAMPException&&) = default;

  virtual const char* what() const noexcept { return _msg.c_str(); }

 protected:
  std::string _msg;
};

#endif  // SCAMP_SCAMP_EXCEPTION_H
