/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#include "uwb_manager.h"

#include <Eigen/Dense>
#include <xmlrpcpp/XmlRpcValue.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

namespace
{
bool xmlRpcHasMember(const XmlRpc::XmlRpcValue &value, const std::string &key)
{
  return value.getType() == XmlRpc::XmlRpcValue::TypeStruct && value.hasMember(key);
}

bool xmlRpcToDouble(const XmlRpc::XmlRpcValue &value, double &out)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
  {
    out = static_cast<double>(value);
    return true;
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    out = static_cast<int>(value);
    return true;
  }
  return false;
}

bool xmlRpcToInt(const XmlRpc::XmlRpcValue &value, int &out)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    out = static_cast<int>(value);
    return true;
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
  {
    const double d = static_cast<double>(value);
    out = static_cast<int>(std::llround(d));
    return std::fabs(d - static_cast<double>(out)) < 1e-6;
  }
  return false;
}

bool xmlRpcToBool(const XmlRpc::XmlRpcValue &value, bool &out)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeBoolean)
  {
    out = static_cast<bool>(value);
    return true;
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    out = (static_cast<int>(value) != 0);
    return true;
  }
  return false;
}

bool xmlRpcToVec3(const XmlRpc::XmlRpcValue &value, V3D &out)
{
  if (value.getType() != XmlRpc::XmlRpcValue::TypeArray || value.size() != 3) return false;
  double x = 0.0, y = 0.0, z = 0.0;
  if (!xmlRpcToDouble(value[0], x) || !xmlRpcToDouble(value[1], y) || !xmlRpcToDouble(value[2], z)) return false;
  out << x, y, z;
  return true;
}

bool xmlRpcGetInt(const XmlRpc::XmlRpcValue &value, const std::string &key, int &out)
{
  return xmlRpcHasMember(value, key) && xmlRpcToInt(value[key], out);
}

bool xmlRpcGetDouble(const XmlRpc::XmlRpcValue &value, const std::string &key, double &out)
{
  return xmlRpcHasMember(value, key) && xmlRpcToDouble(value[key], out);
}

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}

speed_t baudrateToTermios(int baudrate)
{
  switch (baudrate)
  {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
#ifdef B230400
    case 230400: return B230400;
#endif
#ifdef B460800
    case 460800: return B460800;
#endif
#ifdef B921600
    case 921600: return B921600;
#endif
    default: return B115200;
  }
}

std::string trimLine(const std::string &line)
{
  const size_t first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const size_t last = line.find_last_not_of(" \t\r\n");
  return line.substr(first, last - first + 1);
}

bool isIntegerLike(double value)
{
  return std::isfinite(value) && std::fabs(value - std::round(value)) < 1e-6;
}

bool regexFindInt(const std::string &line, const std::regex &regex, int &value)
{
  std::smatch match;
  if (!std::regex_search(line, match, regex) || match.size() < 2) return false;
  try
  {
    value = std::stoi(match[1].str());
    return true;
  }
  catch (const std::exception &)
  {
    return false;
  }
}

bool regexFindDouble(const std::string &line, const std::regex &regex, double &value)
{
  std::smatch match;
  if (!std::regex_search(line, match, regex) || match.size() < 2) return false;
  try
  {
    value = std::stod(match[1].str());
    return true;
  }
  catch (const std::exception &)
  {
    return false;
  }
}

bool lineContainsErrorStatus(const std::string &line)
{
  return line.find("[ABORT]") != std::string::npos ||
         line.find("[ERROR]") != std::string::npos ||
         line.find("ABORT") != std::string::npos ||
         line.find("ERROR") != std::string::npos;
}

std::string toLower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string expandUserPath(const std::string &path)
{
  if (path.size() >= 2 && path[0] == '~' && path[1] == '/')
  {
    const char *home = std::getenv("HOME");
    if (home != nullptr) return std::string(home) + path.substr(1);
  }
  return path;
}
} // namespace

UwbManager::UwbManager() = default;

UwbManager::~UwbManager()
{
  shutdown();
}

bool UwbManager::initialize(ros::NodeHandle &nh, const std::string &save_path)
{
  if (!loadParameters(nh) || !en_) return false;

  const std::string log_path = save_path + log_filename_;
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    log_file_.open(log_path, std::ios::out | std::ios::app);
    if (log_file_.is_open())
    {
      log_file_ << "# stamp raw_line parsed_ranges_m update_info\n";
      log_file_.flush();
    }
    else
    {
      ROS_WARN("[UWB] Failed to open log file: %s", log_path.c_str());
    }
  }

  const std::string source = toLower(input_source_);
  if (source == "file" || source == "txt" || source == "replay")
  {
    if (!loadReplayFile())
    {
      ROS_WARN("[UWB] Replay source is enabled, but replay file could not be loaded.");
      return false;
    }
    ROS_INFO("[UWB] Replay source loaded: file=%s measurements=%zu time_mode=%s speed=%.3f",
             replay_file_.c_str(), replay_measurements_.size(), replay_time_mode_.c_str(), replay_speed_);
    if (update_en_ && anchors_.empty())
    {
      ROS_WARN("[UWB] uwb/update_en is true, but no enabled anchors with positions were configured. Replay data will be parsed only.");
    }
    return true;
  }

  if (!openSerial())
  {
    ROS_WARN("[UWB] Serial reader is disabled because %s could not be opened/configured.", serial_port_.c_str());
    return false;
  }

  running_.store(true);
  read_thread_ = std::thread(&UwbManager::readLoop, this);
  ROS_INFO("[UWB] Serial reader started: port=%s baud=%d DTR=%d RTS=%d log=%s",
           serial_port_.c_str(), baudrate_, static_cast<int>(dtr_high_), static_cast<int>(rts_high_), log_path.c_str());

  if (update_en_ && anchors_.empty())
  {
    ROS_WARN("[UWB] uwb/update_en is true, but no enabled anchors with positions were configured. Data will be logged only.");
  }
  return true;
}

void UwbManager::shutdown()
{
  running_.store(false);
  if (read_thread_.joinable()) read_thread_.join();
  closeSerial();
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.is_open())
    {
      log_file_.flush();
      log_file_.close();
    }
  }
}

bool UwbManager::loadParameters(ros::NodeHandle &nh)
{
  nh.param<bool>("uwb/en", en_, false);
  nh.param<bool>("uwb/update_en", update_en_, true);
  nh.param<std::string>("uwb/source", input_source_, "serial");
  nh.param<std::string>("uwb/serial_port", serial_port_, "/dev/ttyUSB0");
  nh.param<int>("uwb/baudrate", baudrate_, 115200);
  nh.param<bool>("uwb/dtr", dtr_high_, true);
  nh.param<bool>("uwb/rts", rts_high_, false);
  nh.param<std::string>("uwb/parser_mode", parser_mode_, "uwb");
  nh.param<std::string>("uwb/log_filename", log_filename_, "uwb_ranges.txt");
  nh.param<int>("uwb/log_flush_stride", log_flush_stride_, 1);
  nh.param<std::string>("uwb/replay_file", replay_file_, "");
  nh.param<std::string>("uwb/replay_time_mode", replay_time_mode_, "relative");
  nh.param<double>("uwb/replay_speed", replay_speed_, 1.0);
  nh.param<double>("uwb/range_scale", range_scale_, 1.0);
  nh.param<double>("uwb/min_range_m", min_range_m_, 0.05);
  nh.param<double>("uwb/max_range_m", max_range_m_, 100.0);
  nh.param<double>("uwb/max_age_s", max_age_s_, 0.5);
  nh.param<int>("uwb/max_queue_size", max_queue_size_, 512);
  nh.param<int>("uwb/min_update_anchors", min_update_anchors_, 1);
  nh.param<double>("uwb/range_noise_m", range_noise_m_, 0.20);
  nh.param<double>("uwb/max_residual_m", max_residual_m_, 3.0);
  nh.param<double>("uwb/update_max_rot_step_deg", update_max_rot_step_deg_, 1.0);
  nh.param<double>("uwb/update_max_trans_step_m", update_max_trans_step_m_, 0.10);
  nh.param<bool>("uwb/tag_offset_estimate_en", tag_offset_estimate_en_, false);
  nh.param<int>("uwb/tag_offset_estimate_min_anchors", tag_offset_estimate_min_anchors_, 2);
  nh.param<double>("uwb/tag_offset_init_cov_m", tag_offset_init_cov_m_, 0.10);
  nh.param<double>("uwb/tag_offset_process_noise_m", tag_offset_process_noise_m_, 0.0);
  nh.param<double>("uwb/tag_offset_update_max_step_m", tag_offset_update_max_step_m_, 0.01);
  nh.param<double>("uwb/tag_offset_max_norm_m", tag_offset_max_norm_m_, 1.0);
  nh.param<bool>("uwb/anchor_position_estimate_en", anchor_position_estimate_en_, false);
  nh.param<bool>("uwb/anchor_estimate_use_for_update", anchor_estimate_use_for_update_, true);
  nh.param<bool>("uwb/anchor_estimate_freeze_after_init", anchor_estimate_freeze_after_init_, true);
  nh.param<int>("uwb/anchor_estimate_min_samples", anchor_estimate_min_samples_, 30);
  nh.param<int>("uwb/anchor_estimate_max_samples", anchor_estimate_max_samples_, 300);
  nh.param<int>("uwb/anchor_estimate_min_rank", anchor_estimate_min_rank_, 2);
  nh.param<double>("uwb/anchor_estimate_min_motion_m", anchor_estimate_min_motion_m_, 1.0);
  nh.param<double>("uwb/anchor_estimate_max_rmse_m", anchor_estimate_max_rmse_m_, 0.50);
  nh.param<double>("uwb/anchor_estimate_max_step_m", anchor_estimate_max_step_m_, 2.0);

  std::vector<double> tag_offset;
  nh.param<std::vector<double>>("uwb/tag_offset_body", tag_offset, std::vector<double>{0.0, 0.0, 0.0});
  if (tag_offset.size() == 3)
  {
    tag_offset_body_ << tag_offset[0], tag_offset[1], tag_offset[2];
  }
  else
  {
    ROS_WARN("[UWB] uwb/tag_offset_body must have 3 values. Use [0,0,0].");
    tag_offset_body_.setZero();
  }
  tag_offset_est_body_ = tag_offset_body_;

  range_scale_ = std::max(1e-9, range_scale_);
  log_flush_stride_ = std::max(1, log_flush_stride_);
  replay_speed_ = std::max(1e-6, replay_speed_);
  input_source_ = toLower(input_source_);
  replay_time_mode_ = toLower(replay_time_mode_);
  min_range_m_ = std::max(0.0, min_range_m_);
  max_range_m_ = std::max(min_range_m_, max_range_m_);
  max_age_s_ = std::max(0.0, max_age_s_);
  max_queue_size_ = std::max(8, max_queue_size_);
  min_update_anchors_ = std::max(1, min_update_anchors_);
  range_noise_m_ = std::max(1e-3, range_noise_m_);
  max_residual_m_ = std::max(0.0, max_residual_m_);
  update_max_rot_step_deg_ = std::max(0.0, update_max_rot_step_deg_);
  update_max_trans_step_m_ = std::max(0.0, update_max_trans_step_m_);
  tag_offset_estimate_min_anchors_ = std::max(1, tag_offset_estimate_min_anchors_);
  tag_offset_init_cov_m_ = std::max(1e-4, tag_offset_init_cov_m_);
  tag_offset_process_noise_m_ = std::max(0.0, tag_offset_process_noise_m_);
  tag_offset_update_max_step_m_ = std::max(0.0, tag_offset_update_max_step_m_);
  tag_offset_max_norm_m_ = std::max(0.0, tag_offset_max_norm_m_);
  tag_offset_cov_ = M3D::Identity() * (tag_offset_init_cov_m_ * tag_offset_init_cov_m_);
  anchor_estimate_min_samples_ = std::max(4, anchor_estimate_min_samples_);
  anchor_estimate_max_samples_ = std::max(anchor_estimate_min_samples_, anchor_estimate_max_samples_);
  anchor_estimate_min_rank_ = std::max(1, std::min(3, anchor_estimate_min_rank_));
  anchor_estimate_min_motion_m_ = std::max(0.0, anchor_estimate_min_motion_m_);
  anchor_estimate_max_rmse_m_ = std::max(0.0, anchor_estimate_max_rmse_m_);
  anchor_estimate_max_step_m_ = std::max(0.0, anchor_estimate_max_step_m_);

  XmlRpc::XmlRpcValue anchors_param;
  if (nh.getParam("uwb/anchors", anchors_param) && anchors_param.getType() == XmlRpc::XmlRpcValue::TypeArray)
  {
    for (int i = 0; i < anchors_param.size(); ++i)
    {
      XmlRpc::XmlRpcValue anchor_param = anchors_param[i];
      if (anchor_param.getType() != XmlRpc::XmlRpcValue::TypeStruct) continue;

      UwbAnchor anchor;
      anchor.id = i;
      if (xmlRpcHasMember(anchor_param, "id"))
      {
        int id = i;
        if (xmlRpcToInt(anchor_param["id"], id)) anchor.id = id;
      }

      bool enabled = true;
      if (xmlRpcHasMember(anchor_param, "enable")) xmlRpcToBool(anchor_param["enable"], enabled);
      if (xmlRpcHasMember(anchor_param, "en")) xmlRpcToBool(anchor_param["en"], enabled);
      if (xmlRpcHasMember(anchor_param, "flag")) xmlRpcToBool(anchor_param["flag"], enabled);
      anchor.enabled = enabled;

      bool has_position = false;
      if (xmlRpcHasMember(anchor_param, "position"))
      {
        has_position = xmlRpcToVec3(anchor_param["position"], anchor.position_w);
      }
      if (!has_position)
      {
        ROS_WARN("[UWB] Anchor %d has no valid position. It can be parsed but will not be used for EKF update.", anchor.id);
        anchor.enabled = false;
      }

      anchor_order_.push_back(anchor.id);
      configured_anchors_[anchor.id] = anchor;
      if (anchor.enabled) anchors_[anchor.id] = anchor;
    }
  }

  XmlRpc::XmlRpcValue constraints_param;
  if (nh.getParam("uwb/anchor_distance_constraints", constraints_param) &&
      constraints_param.getType() == XmlRpc::XmlRpcValue::TypeArray)
  {
    for (int i = 0; i < constraints_param.size(); ++i)
    {
      XmlRpc::XmlRpcValue constraint_param = constraints_param[i];
      if (constraint_param.getType() != XmlRpc::XmlRpcValue::TypeStruct) continue;

      UwbAnchorDistanceConstraint constraint;
      bool ok = xmlRpcGetInt(constraint_param, "from", constraint.id_a) ||
                xmlRpcGetInt(constraint_param, "id_a", constraint.id_a);
      ok = (xmlRpcGetInt(constraint_param, "to", constraint.id_b) ||
            xmlRpcGetInt(constraint_param, "id_b", constraint.id_b)) && ok;
      ok = xmlRpcGetDouble(constraint_param, "distance", constraint.distance_m) && ok;
      if (!ok || constraint.id_a == constraint.id_b || constraint.distance_m <= 0.0)
      {
        ROS_WARN("[UWB] Skip invalid anchor distance constraint at index %d.", i);
        continue;
      }
      anchor_distance_constraints_.push_back(constraint);
    }
  }

  ROS_INFO("[UWB] enable=%d update=%d parser=%s anchors_for_update=%zu",
           static_cast<int>(en_), static_cast<int>(update_en_), parser_mode_.c_str(), anchors_.size());
  if (tag_offset_estimate_en_)
  {
    ROS_WARN("[UWB] Online tag_offset estimation is enabled. Use it only with fixed anchors, enough motion, and checked range outliers.");
  }
  if (anchor_position_estimate_en_)
  {
    ROS_WARN("[UWB] Online anchor position estimation is enabled for flag=0 anchors. Estimates need enough trajectory excitation; tunnel-line motion may not fully constrain 3D.");
  }
  return true;
}

bool UwbManager::openSerial()
{
  serial_fd_ = ::open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (serial_fd_ < 0)
  {
    ROS_WARN("[UWB] Failed to open %s: %s", serial_port_.c_str(), std::strerror(errno));
    return false;
  }

  if (!configureSerial())
  {
    closeSerial();
    return false;
  }
  return true;
}

bool UwbManager::configureSerial()
{
  if (serial_fd_ < 0) return false;

  termios tty;
  if (::tcgetattr(serial_fd_, &tty) != 0)
  {
    ROS_WARN("[UWB] tcgetattr failed on %s: %s", serial_port_.c_str(), std::strerror(errno));
    return false;
  }

  cfmakeraw(&tty);
  const speed_t baud = baudrateToTermios(baudrate_);
  ::cfsetispeed(&tty, baud);
  ::cfsetospeed(&tty, baud);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;
  tty.c_cflag &= ~PARENB;
  tty.c_cflag &= ~CSTOPB;
#ifdef CRTSCTS
  tty.c_cflag &= ~CRTSCTS;
#endif
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 1;

  if (::tcsetattr(serial_fd_, TCSANOW, &tty) != 0)
  {
    ROS_WARN("[UWB] tcsetattr failed on %s: %s", serial_port_.c_str(), std::strerror(errno));
    return false;
  }
  ::tcflush(serial_fd_, TCIOFLUSH);

  int modem_bits = 0;
  if (::ioctl(serial_fd_, TIOCMGET, &modem_bits) == 0)
  {
    if (dtr_high_) modem_bits |= TIOCM_DTR;
    else modem_bits &= ~TIOCM_DTR;
    if (rts_high_) modem_bits |= TIOCM_RTS;
    else modem_bits &= ~TIOCM_RTS;
    if (::ioctl(serial_fd_, TIOCMSET, &modem_bits) != 0)
    {
      ROS_WARN("[UWB] Failed to set DTR/RTS on %s: %s", serial_port_.c_str(), std::strerror(errno));
    }
  }
  else
  {
    ROS_WARN("[UWB] Failed to read modem bits on %s: %s", serial_port_.c_str(), std::strerror(errno));
  }

  return true;
}

void UwbManager::closeSerial()
{
  if (serial_fd_ >= 0)
  {
    ::close(serial_fd_);
    serial_fd_ = -1;
  }
}

void UwbManager::readLoop()
{
  std::string line_buffer;
  line_buffer.reserve(512);
  char buffer[256];

  while (running_.load())
  {
    const ssize_t n = ::read(serial_fd_, buffer, sizeof(buffer));
    if (n > 0)
    {
      for (ssize_t i = 0; i < n; ++i)
      {
        const char c = buffer[i];
        if (c == '\n' || c == '\r')
        {
          const std::string line = trimLine(line_buffer);
          line_buffer.clear();
          if (!line.empty()) handleLine(line, ros::Time::now().toSec());
        }
        else
        {
          line_buffer.push_back(c);
          if (line_buffer.size() > 4096)
          {
            const std::string line = trimLine(line_buffer);
            line_buffer.clear();
            if (!line.empty()) handleLine(line, ros::Time::now().toSec());
          }
        }
      }
    }
    else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
    {
      ROS_WARN_THROTTLE(5.0, "[UWB] Serial read error on %s: %s", serial_port_.c_str(), std::strerror(errno));
      ros::Duration(0.02).sleep();
    }
    else
    {
      ros::Duration(0.005).sleep();
    }
  }
}

bool UwbManager::loadReplayFile()
{
  if (replay_file_.empty())
  {
    ROS_WARN("[UWB] uwb/replay_file is empty.");
    return false;
  }

  replay_file_ = expandUserPath(replay_file_);
  std::ifstream replay_file(replay_file_);
  if (!replay_file.is_open())
  {
    ROS_WARN("[UWB] Failed to open replay file: %s", replay_file_.c_str());
    return false;
  }

  replay_measurements_.clear();
  replay_index_ = 0;
  replay_started_ = false;

  std::string line;
  while (std::getline(replay_file, line))
  {
    line = trimLine(line);
    if (line.empty() || line[0] == '#') continue;
    if (line.find(" UPDATE ") != std::string::npos || line.find("UPDATE used=") != std::string::npos) continue;

    std::istringstream iss(line);
    double stamp = 0.0;
    if (!(iss >> stamp)) continue;

    std::string raw_line;
    const std::string raw_key = "raw=\"";
    const size_t raw_begin = line.find(raw_key);
    if (raw_begin != std::string::npos)
    {
      const size_t content_begin = raw_begin + raw_key.size();
      const size_t content_end = line.find('"', content_begin);
      if (content_end != std::string::npos)
      {
        raw_line = line.substr(content_begin, content_end - content_begin);
      }
    }

    if (raw_line.empty())
    {
      const size_t first_space = line.find_first_of(" \t");
      raw_line = first_space == std::string::npos ? line : trimLine(line.substr(first_space + 1));
    }

    auto measurements = parseLine(raw_line, stamp);
    for (auto &measurement : measurements)
    {
      measurement.stamp = stamp;
      replay_measurements_.push_back(measurement);
    }
  }

  std::sort(replay_measurements_.begin(), replay_measurements_.end(),
            [](const UwbRangeMeasurement &a, const UwbRangeMeasurement &b) {
              return a.stamp < b.stamp;
            });

  if (!replay_measurements_.empty())
  {
    replay_file_start_stamp_ = replay_measurements_.front().stamp;
  }
  return !replay_measurements_.empty();
}

std::vector<UwbRangeMeasurement> UwbManager::takeReplayMeasurements(double now)
{
  std::vector<UwbRangeMeasurement> measurements;
  if (replay_measurements_.empty() || replay_index_ >= replay_measurements_.size()) return measurements;
  if (now <= 0.0) return measurements;

  if (!replay_started_)
  {
    replay_started_ = true;
    replay_ros_start_stamp_ = now;
    replay_file_start_stamp_ = replay_measurements_.front().stamp;
  }

  while (replay_index_ < replay_measurements_.size())
  {
    const auto &measurement = replay_measurements_[replay_index_];
    double due_time = measurement.stamp;
    if (replay_time_mode_ != "absolute")
    {
      due_time = replay_ros_start_stamp_ + (measurement.stamp - replay_file_start_stamp_) / replay_speed_;
    }

    if (due_time > now) break;
    measurements.push_back(measurement);
    replay_index_++;
  }
  return measurements;
}

void UwbManager::handleLine(const std::string &line, double stamp)
{
  const auto measurements = parseLine(line, stamp);
  logRawLine(stamp, line, measurements);
  if (measurements.empty()) return;

  std::lock_guard<std::mutex> lock(measurement_mutex_);
  for (const auto &measurement : measurements)
  {
    measurement_queue_.push_back(measurement);
  }
  while (static_cast<int>(measurement_queue_.size()) > max_queue_size_)
  {
    measurement_queue_.pop_front();
  }
}

std::vector<UwbRangeMeasurement> UwbManager::parseLine(const std::string &line, double stamp) const
{
  std::vector<UwbRangeMeasurement> measurements;

  auto appendMeasurement = [&](int anchor_id, double raw_range)
  {
    const double range_m = raw_range * range_scale_;
    if (!std::isfinite(range_m) || range_m <= 0.0 || range_m < min_range_m_ || range_m > max_range_m_) return;
    UwbRangeMeasurement measurement;
    measurement.anchor_id = anchor_id;
    measurement.range_m = range_m;
    measurement.stamp = stamp;
    measurement.raw_line = line;
    measurements.push_back(measurement);
  };

  static const std::regex distance_line_regex(
      R"(distance\s*\[\s*([-+]?\d+)\s*\]\s*,\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))",
      std::regex::icase);
  std::smatch distance_match;
  if (std::regex_search(line, distance_match, distance_line_regex))
  {
    if (!lineContainsErrorStatus(line))
    {
      appendMeasurement(std::stoi(distance_match[1].str()), std::stod(distance_match[2].str()));
    }
    return measurements;
  }

  static const std::regex target_regex(R"(\btarget\s*=\s*([-+]?\d+))", std::regex::icase);
  static const std::regex ok_regex(R"(\bok\s*=\s*([-+]?\d+))", std::regex::icase);
  static const std::regex dist_regex(R"(\bdist\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))", std::regex::icase);
  if (line.find("[UWBDBG]") != std::string::npos || line.find("dist=") != std::string::npos)
  {
    int target_id = -1;
    int ok = 0;
    double dist_m = 0.0;
    if (regexFindInt(line, target_regex, target_id) &&
        regexFindInt(line, ok_regex, ok) &&
        regexFindDouble(line, dist_regex, dist_m) &&
        ok == 1 &&
        !lineContainsErrorStatus(line))
    {
      appendMeasurement(target_id, dist_m);
    }
    return measurements;
  }

  if (parser_mode_ == "uwb" || parser_mode_ == "distance") return measurements;

  std::vector<double> values;
  static const std::regex number_regex(R"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)");
  auto begin = std::sregex_iterator(line.begin(), line.end(), number_regex);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it)
  {
    try
    {
      values.push_back(std::stod(it->str()));
    }
    catch (const std::exception &)
    {
    }
  }
  if (values.empty()) return measurements;

  bool parse_as_pairs = (parser_mode_ == "pairs");
  if (parser_mode_ == "auto" && values.size() >= 2 && values.size() % 2 == 0)
  {
    parse_as_pairs = true;
    for (size_t i = 0; i < values.size(); i += 2)
    {
      if (!isIntegerLike(values[i]))
      {
        parse_as_pairs = false;
        break;
      }
      const int id = static_cast<int>(std::llround(values[i]));
      if (!anchor_order_.empty() &&
          std::find(anchor_order_.begin(), anchor_order_.end(), id) == anchor_order_.end())
      {
        parse_as_pairs = false;
        break;
      }
    }
  }

  if (parse_as_pairs)
  {
    for (size_t i = 0; i + 1 < values.size(); i += 2)
    {
      if (!isIntegerLike(values[i])) continue;
      appendMeasurement(static_cast<int>(std::llround(values[i])), values[i + 1]);
    }
    return measurements;
  }

  for (size_t i = 0; i < values.size(); ++i)
  {
    const int anchor_id = (i < anchor_order_.size()) ? anchor_order_[i] : static_cast<int>(i);
    appendMeasurement(anchor_id, values[i]);
  }
  return measurements;
}

std::vector<UwbRangeMeasurement> UwbManager::takeRecentMeasurements(double now)
{
  std::vector<UwbRangeMeasurement> measurements;
  std::lock_guard<std::mutex> lock(measurement_mutex_);

  while (!measurement_queue_.empty())
  {
    UwbRangeMeasurement measurement = measurement_queue_.front();
    measurement_queue_.pop_front();
    if (max_age_s_ > 0.0 && now - measurement.stamp > max_age_s_) continue;
    measurements.push_back(measurement);
  }
  return measurements;
}

void UwbManager::logRawLine(double stamp, const std::string &line, const std::vector<UwbRangeMeasurement> &measurements)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!log_file_.is_open()) return;

  log_file_ << std::fixed << std::setprecision(6) << stamp << " raw=\"" << line << "\" parsed=";
  if (measurements.empty())
  {
    log_file_ << "none";
  }
  else
  {
    for (const auto &measurement : measurements)
    {
      log_file_ << measurement.anchor_id << ":" << std::setprecision(4) << measurement.range_m << "m ";
    }
  }
  log_file_ << '\n';
  log_pending_lines_++;
  if (log_pending_lines_ >= log_flush_stride_)
  {
    log_file_.flush();
    log_pending_lines_ = 0;
  }
}

void UwbManager::logAnchorEstimate(int anchor_id, const V3D &position_w, double rmse, int rank, int sample_count)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!log_file_.is_open()) return;

  log_file_ << std::fixed << std::setprecision(6)
            << ros::Time::now().toSec()
            << " ANCHOR_ESTIMATE id=" << anchor_id
            << " position=" << position_w.transpose()
            << " rmse=" << rmse
            << " rank=" << rank
            << " samples=" << sample_count
            << '\n';
  log_file_.flush();
}

void UwbManager::collectAnchorEstimateSamples(const StatesGroup &state, const std::vector<UwbRangeMeasurement> &measurements)
{
  if (!anchor_position_estimate_en_) return;

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;

  for (const auto &measurement : measurements)
  {
    if (measurement.range_m <= 0.0) continue;

    const auto known_anchor_it = anchors_.find(measurement.anchor_id);
    if (known_anchor_it != anchors_.end() &&
        (!known_anchor_it->second.estimated || anchor_estimate_freeze_after_init_))
    {
      continue;
    }

    UwbAnchorSample sample;
    sample.tag_position_w = tag_position_w;
    sample.range_m = measurement.range_m;
    sample.stamp = measurement.stamp;
    auto &samples = anchor_samples_[measurement.anchor_id];
    samples.push_back(sample);
    while (static_cast<int>(samples.size()) > anchor_estimate_max_samples_)
    {
      samples.pop_front();
    }
  }

  for (auto &item : anchor_samples_)
  {
    const int anchor_id = item.first;
    const auto existing_anchor_it = anchors_.find(anchor_id);
    if (existing_anchor_it != anchors_.end() && anchor_estimate_freeze_after_init_) continue;

    UwbAnchor estimated_anchor;
    double rmse = 0.0;
    int rank = 0;
    if (!estimateAnchorPosition(anchor_id, estimated_anchor, rmse, rank)) continue;

    configured_anchors_[anchor_id] = estimated_anchor;
    if (anchor_estimate_use_for_update_)
    {
      anchors_[anchor_id] = estimated_anchor;
      applyAnchorDistanceConstraints();
    }

    const V3D logged_position =
        (anchor_estimate_use_for_update_ && anchors_.find(anchor_id) != anchors_.end()) ?
        anchors_[anchor_id].position_w : estimated_anchor.position_w;
    logAnchorEstimate(anchor_id, logged_position, rmse, rank,
                      static_cast<int>(item.second.size()));
    ROS_INFO("[UWB] Estimated anchor %d position=[%.3f %.3f %.3f], rmse=%.3f m, rank=%d, samples=%zu, use_for_update=%d",
             anchor_id,
             logged_position.x(),
             logged_position.y(),
             logged_position.z(),
             rmse,
             rank,
             item.second.size(),
             static_cast<int>(anchor_estimate_use_for_update_));
  }
}

bool UwbManager::estimateAnchorPosition(int anchor_id, UwbAnchor &anchor, double &rmse, int &rank) const
{
  const auto samples_it = anchor_samples_.find(anchor_id);
  if (samples_it == anchor_samples_.end()) return false;

  const auto &samples = samples_it->second;
  const int n = static_cast<int>(samples.size());
  if (n < anchor_estimate_min_samples_) return false;

  V3D min_p = samples.front().tag_position_w;
  V3D max_p = samples.front().tag_position_w;
  for (const auto &sample : samples)
  {
    min_p = min_p.cwiseMin(sample.tag_position_w);
    max_p = max_p.cwiseMax(sample.tag_position_w);
  }
  if ((max_p - min_p).norm() < anchor_estimate_min_motion_m_) return false;

  const auto &ref = samples.front();
  Eigen::MatrixXd A(n - 1, 3);
  Eigen::VectorXd b(n - 1);
  for (int i = 1; i < n; ++i)
  {
    const auto &sample = samples[i];
    A.row(i - 1) = 2.0 * (ref.tag_position_w - sample.tag_position_w).transpose();
    b(i - 1) = sample.range_m * sample.range_m - ref.range_m * ref.range_m -
               sample.tag_position_w.squaredNorm() + ref.tag_position_w.squaredNorm();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.singularValues().size() == 0) return false;
  const double max_sv = svd.singularValues()(0);
  rank = 0;
  for (int i = 0; i < svd.singularValues().size(); ++i)
  {
    if (svd.singularValues()(i) > std::max(1e-6, max_sv * 1e-3)) rank++;
  }
  if (rank < anchor_estimate_min_rank_) return false;

  V3D estimate = svd.solve(b);
  if (!estimate.allFinite()) return false;

  for (int iter = 0; iter < 8; ++iter)
  {
    Eigen::MatrixXd H(n, 3);
    Eigen::VectorXd residual(n);
    for (int i = 0; i < n; ++i)
    {
      const V3D diff = estimate - samples[i].tag_position_w;
      const double predicted = diff.norm();
      if (predicted < 1e-6) return false;
      H.row(i) = (diff / predicted).transpose();
      residual(i) = samples[i].range_m - predicted;
    }

    Eigen::Matrix3d normal = H.transpose() * H + Eigen::Matrix3d::Identity() * 1e-6;
    V3D delta = normal.ldlt().solve(H.transpose() * residual);
    if (!delta.allFinite()) return false;
    if (anchor_estimate_max_step_m_ > 0.0 && delta.norm() > anchor_estimate_max_step_m_)
    {
      delta = delta.normalized() * anchor_estimate_max_step_m_;
    }
    estimate += delta;
    if (delta.norm() < 1e-4) break;
  }

  double residual_sum = 0.0;
  for (const auto &sample : samples)
  {
    const double residual = sample.range_m - (estimate - sample.tag_position_w).norm();
    residual_sum += residual * residual;
  }
  rmse = std::sqrt(residual_sum / std::max(1, n));
  if (anchor_estimate_max_rmse_m_ > 0.0 && rmse > anchor_estimate_max_rmse_m_) return false;

  anchor.id = anchor_id;
  anchor.enabled = true;
  anchor.estimated = true;
  anchor.position_w = estimate;
  return true;
}

void UwbManager::applyAnchorDistanceConstraints()
{
  for (const auto &constraint : anchor_distance_constraints_)
  {
    auto a_it = anchors_.find(constraint.id_a);
    auto b_it = anchors_.find(constraint.id_b);
    if (a_it == anchors_.end() || b_it == anchors_.end()) continue;

    V3D diff = b_it->second.position_w - a_it->second.position_w;
    const double current_distance = diff.norm();
    if (current_distance < 1e-6) continue;

    const V3D dir = diff / current_distance;
    if (a_it->second.estimated && b_it->second.estimated)
    {
      const V3D midpoint = 0.5 * (a_it->second.position_w + b_it->second.position_w);
      a_it->second.position_w = midpoint - 0.5 * constraint.distance_m * dir;
      b_it->second.position_w = midpoint + 0.5 * constraint.distance_m * dir;
    }
    else if (a_it->second.estimated)
    {
      a_it->second.position_w = b_it->second.position_w - constraint.distance_m * dir;
    }
    else if (b_it->second.estimated)
    {
      b_it->second.position_w = a_it->second.position_w + constraint.distance_m * dir;
    }
  }
}

void UwbManager::logUpdate(double stamp, int used_count, double residual_norm, const V3D &rot_add,
                           const V3D &trans_add, const V3D &tag_offset_add)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!log_file_.is_open()) return;

  log_file_ << std::fixed << std::setprecision(6)
            << stamp << " UPDATE used=" << used_count
            << " residual_norm=" << residual_norm
            << " rot_add=" << rot_add.transpose()
            << " trans_add=" << trans_add.transpose()
            << " tag_offset=" << tag_offset_est_body_.transpose()
            << " tag_offset_add=" << tag_offset_add.transpose()
            << '\n';
  log_file_.flush();
}

int UwbManager::applyRangeUpdate(StatesGroup &state)
{
  if (!en_ || !update_en_) return 0;
  const double now = ros::Time::now().toSec();
  const std::string source = toLower(input_source_);
  const auto measurements = (source == "file" || source == "txt" || source == "replay") ?
                            takeReplayMeasurements(now) :
                            takeRecentMeasurements(now);
  if (measurements.empty()) return 0;
  collectAnchorEstimateSamples(state, measurements);
  if (anchors_.empty()) return 0;
  return applyLatestMeasurements(state, measurements);
}

int UwbManager::applyLatestMeasurements(StatesGroup &state, const std::vector<UwbRangeMeasurement> &measurements)
{
  std::map<int, UwbRangeMeasurement> latest_by_anchor;
  for (const auto &measurement : measurements)
  {
    if (anchors_.find(measurement.anchor_id) == anchors_.end()) continue;
    latest_by_anchor[measurement.anchor_id] = measurement;
  }
  if (static_cast<int>(latest_by_anchor.size()) < min_update_anchors_) return 0;

  std::vector<UwbRangeMeasurement> usable_measurements;
  usable_measurements.reserve(latest_by_anchor.size());
  for (const auto &item : latest_by_anchor)
  {
    usable_measurements.push_back(item.second);
  }

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(usable_measurements.size(), DIM_STATE);
  Eigen::MatrixXd H_tag = Eigen::MatrixXd::Zero(usable_measurements.size(), 3);
  Eigen::VectorXd z = Eigen::VectorXd::Zero(usable_measurements.size());

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;
  int row = 0;
  for (const auto &measurement : usable_measurements)
  {
    const auto anchor_it = anchors_.find(measurement.anchor_id);
    if (anchor_it == anchors_.end()) continue;

    const V3D diff = tag_position_w - anchor_it->second.position_w;
    const double predicted_range = diff.norm();
    if (predicted_range < 1e-6) continue;

    const double residual = measurement.range_m - predicted_range;
    if (!std::isfinite(residual)) continue;
    if (max_residual_m_ > 0.0 && std::fabs(residual) > max_residual_m_)
    {
      ROS_WARN_THROTTLE(2.0, "[UWB] Reject range anchor=%d residual=%.3f m > %.3f m",
                        measurement.anchor_id, residual, max_residual_m_);
      continue;
    }

    const V3D direction = diff / predicted_range;
    H.block<1, 3>(row, 0) = direction.transpose() * (-state.rot_end * skewSymmetric(tag_offset_used));
    H.block<1, 3>(row, 3) = direction.transpose();
    if (tag_offset_estimate_en_)
    {
      H_tag.block<1, 3>(row, 0) = direction.transpose() * state.rot_end;
    }
    z(row) = residual;
    row++;
  }

  if (row < min_update_anchors_) return 0;
  H.conservativeResize(row, DIM_STATE);
  H_tag.conservativeResize(row, 3);
  z.conservativeResize(row);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(row, row) * (range_noise_m_ * range_noise_m_);

  const bool estimate_tag_offset_this_update =
      tag_offset_estimate_en_ && row >= tag_offset_estimate_min_anchors_;

  if (!estimate_tag_offset_this_update)
  {
    if (tag_offset_estimate_en_)
    {
      ROS_WARN_THROTTLE(3.0,
                        "[UWB] Skip tag_offset estimation in this update: used anchors=%d < required=%d. Pose update still runs.",
                        row, tag_offset_estimate_min_anchors_);
    }

    const Eigen::MatrixXd S = H * state.cov * H.transpose() + R;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success)
    {
      ROS_WARN_THROTTLE(2.0, "[UWB] Skip update: innovation covariance decomposition failed.");
      return 0;
    }

    const Eigen::MatrixXd K = state.cov * H.transpose() * ldlt.solve(Eigen::MatrixXd::Identity(row, row));
    Eigen::VectorXd dx_dynamic = K * z;
    if (dx_dynamic.size() != DIM_STATE || !dx_dynamic.allFinite()) return 0;

    VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
    dx = dx_dynamic;

    V3D rot_add = dx.block<3, 1>(0, 0);
    V3D trans_add = dx.block<3, 1>(3, 0);
    const double rot_step_deg = rot_add.norm() * 57.29577951308232;
    const double trans_step_m = trans_add.norm();
    double step_scale = 1.0;
    if (update_max_rot_step_deg_ > 0.0 && rot_step_deg > update_max_rot_step_deg_)
    {
      step_scale = std::min(step_scale, update_max_rot_step_deg_ / std::max(rot_step_deg, 1e-9));
    }
    if (update_max_trans_step_m_ > 0.0 && trans_step_m > update_max_trans_step_m_)
    {
      step_scale = std::min(step_scale, update_max_trans_step_m_ / std::max(trans_step_m, 1e-9));
    }
    if (step_scale < 1.0)
    {
      dx *= step_scale;
      rot_add = dx.block<3, 1>(0, 0);
      trans_add = dx.block<3, 1>(3, 0);
    }

    state += dx;
    const MD(DIM_STATE, DIM_STATE) I_STATE = MD(DIM_STATE, DIM_STATE)::Identity();
    const MD(DIM_STATE, DIM_STATE) I_KH = I_STATE - K * H;
    state.cov = I_KH * state.cov * I_KH.transpose() + K * R * K.transpose();
    state.cov = 0.5 * (state.cov + state.cov.transpose());
    snapStateForDeterminism(state);

    logUpdate(ros::Time::now().toSec(), row, z.norm(), rot_add, trans_add, V3D::Zero());
    ROS_INFO_THROTTLE(1.0, "[UWB] EKF update used=%d residual_norm=%.3f trans_add=%.4f m",
                      row, z.norm(), trans_add.norm());
    return row;
  }

  constexpr int DIM_UWB_JOINT = DIM_STATE + 3;
  Eigen::MatrixXd H_joint = Eigen::MatrixXd::Zero(row, DIM_UWB_JOINT);
  H_joint.block(0, 0, row, DIM_STATE) = H;
  H_joint.block(0, DIM_STATE, row, 3) = H_tag;

  Eigen::MatrixXd P_joint = Eigen::MatrixXd::Zero(DIM_UWB_JOINT, DIM_UWB_JOINT);
  P_joint.block(0, 0, DIM_STATE, DIM_STATE) = state.cov;
  const double tag_process_var = tag_offset_process_noise_m_ * tag_offset_process_noise_m_;
  P_joint.block(DIM_STATE, DIM_STATE, 3, 3) =
      tag_offset_cov_ + M3D::Identity() * tag_process_var;

  const Eigen::MatrixXd S = H_joint * P_joint * H_joint.transpose() + R;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success)
  {
    ROS_WARN_THROTTLE(2.0, "[UWB] Skip update: innovation covariance decomposition failed.");
    return 0;
  }

  const Eigen::MatrixXd K = P_joint * H_joint.transpose() * ldlt.solve(Eigen::MatrixXd::Identity(row, row));
  Eigen::VectorXd dx_dynamic = K * z;
  if (dx_dynamic.size() != DIM_UWB_JOINT || !dx_dynamic.allFinite()) return 0;

  VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
  dx = dx_dynamic.head(DIM_STATE);
  V3D tag_offset_add = dx_dynamic.segment<3>(DIM_STATE);

  V3D rot_add = dx.block<3, 1>(0, 0);
  V3D trans_add = dx.block<3, 1>(3, 0);
  const double rot_step_deg = rot_add.norm() * 57.29577951308232;
  const double trans_step_m = trans_add.norm();
  double step_scale = 1.0;
  if (update_max_rot_step_deg_ > 0.0 && rot_step_deg > update_max_rot_step_deg_)
  {
    step_scale = std::min(step_scale, update_max_rot_step_deg_ / std::max(rot_step_deg, 1e-9));
  }
  if (update_max_trans_step_m_ > 0.0 && trans_step_m > update_max_trans_step_m_)
  {
    step_scale = std::min(step_scale, update_max_trans_step_m_ / std::max(trans_step_m, 1e-9));
  }
  if (tag_offset_update_max_step_m_ > 0.0 && tag_offset_add.norm() > tag_offset_update_max_step_m_)
  {
    step_scale = std::min(step_scale, tag_offset_update_max_step_m_ / std::max(tag_offset_add.norm(), 1e-9));
  }
  if (step_scale < 1.0)
  {
    dx *= step_scale;
    tag_offset_add *= step_scale;
    rot_add = dx.block<3, 1>(0, 0);
    trans_add = dx.block<3, 1>(3, 0);
  }

  state += dx;
  tag_offset_est_body_ += tag_offset_add;
  if (tag_offset_max_norm_m_ > 0.0 && tag_offset_est_body_.norm() > tag_offset_max_norm_m_)
  {
    tag_offset_est_body_ = tag_offset_est_body_.normalized() * tag_offset_max_norm_m_;
  }

  const Eigen::MatrixXd I_JOINT = Eigen::MatrixXd::Identity(DIM_UWB_JOINT, DIM_UWB_JOINT);
  const Eigen::MatrixXd I_KH = I_JOINT - K * H_joint;
  Eigen::MatrixXd P_joint_updated = I_KH * P_joint * I_KH.transpose() + K * R * K.transpose();
  P_joint_updated = 0.5 * (P_joint_updated + P_joint_updated.transpose());
  state.cov = P_joint_updated.block(0, 0, DIM_STATE, DIM_STATE);
  tag_offset_cov_ = P_joint_updated.block(DIM_STATE, DIM_STATE, 3, 3);
  snapStateForDeterminism(state);

  logUpdate(ros::Time::now().toSec(), row, z.norm(), rot_add, trans_add, tag_offset_add);
  ROS_INFO_THROTTLE(1.0, "[UWB] EKF update used=%d residual_norm=%.3f trans_add=%.4f m tag_offset=[%.3f %.3f %.3f]",
                    row, z.norm(), trans_add.norm(),
                    tag_offset_est_body_.x(), tag_offset_est_body_.y(), tag_offset_est_body_.z());
  return row;
}
