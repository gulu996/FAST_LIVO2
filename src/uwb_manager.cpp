/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#include "uwb_manager.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
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
#include <limits>
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
      log_file_ << "# stamp raw_line parsed_ranges_m update_info_or_event\n";
      log_file_.flush();
    }
    else
    {
      ROS_WARN("[UWB] Failed to open log file: %s", log_path.c_str());
    }
  }
  if (start_anchor_origin_en_)
  {
    std::ostringstream oss;
    oss << "START_ANCHOR_ORIGIN_DIRECT origin_id=" << start_anchor_origin_id_
        << " anchors_for_update=" << anchors_.size()
        << " tolerance=" << start_anchor_origin_tolerance_m_;
    logEvent(ros::Time::now().toSec(), "INFO", oss.str());
    for (const auto &item : anchors_)
    {
      std::ostringstream anchor_oss;
      anchor_oss << "START_ANCHOR_ORIGIN_ANCHOR id=" << item.first
                 << " position=" << item.second.position_w.transpose();
      logEvent(ros::Time::now().toSec(), "INFO", anchor_oss.str());
    }
  }

  const std::string source = toLower(input_source_);
  if (source == "file" || source == "txt" || source == "replay")
  {
    if (!loadReplayFile())
    {
      ROS_WARN("[UWB] Replay source is enabled, but replay file could not be loaded.");
      logEvent(ros::Time::now().toSec(), "WARN", "REPLAY_LOAD_FAILED file=" + replay_file_);
      return false;
    }
    const bool replay_follows_ros_clock = ros::Time::isSimTime();
    const double effective_replay_speed = replay_follows_ros_clock ? 1.0 : replay_speed_;
    ROS_INFO("[UWB] Replay source loaded: file=%s measurements=%zu time_mode=%s requested_speed=%.3f effective_speed=%.3f sim_time=%d",
             replay_file_.c_str(), replay_measurements_.size(), replay_time_mode_.c_str(),
             replay_speed_, effective_replay_speed, static_cast<int>(replay_follows_ros_clock));
    if (replay_follows_ros_clock && std::fabs(replay_speed_ - 1.0) > 1e-6)
    {
      ROS_WARN("[UWB] Ignoring uwb/replay_speed=%.3f because /use_sim_time is enabled. rosbag --clock already controls replay timing.",
               replay_speed_);
    }
    {
      std::ostringstream oss;
      oss << "REPLAY_LOADED file=" << replay_file_
          << " measurements=" << replay_measurements_.size()
          << " file_start_stamp=" << replay_file_start_stamp_
          << " first_measurement_stamp=" << (replay_measurements_.empty() ? 0.0 : replay_measurements_.front().stamp)
          << " time_mode=" << replay_time_mode_
          << " requested_speed=" << replay_speed_
          << " effective_speed=" << effective_replay_speed
          << " sim_time=" << static_cast<int>(replay_follows_ros_clock);
      logEvent(ros::Time::now().toSec(), "INFO", oss.str());
    }
    if (update_en_ && anchors_.empty())
    {
      if (anchor_frame_align_en_)
      {
        ROS_WARN("[UWB] Configured anchors are in an external frame. Anchor frame alignment will transform them before EKF updates start.");
        logEvent(ros::Time::now().toSec(), "WARN",
                 "NO_LOCAL_ANCHORS anchor_frame_align source=replay");
      }
      else if (baseline_anchor_init_en_)
      {
        ROS_WARN("[UWB] No fixed anchors were configured. Baseline anchor initialization will use start/end ids and known distance before EKF updates start.");
        logEvent(ros::Time::now().toSec(), "WARN",
                 "NO_FIXED_ANCHORS baseline_anchor_init source=replay");
      }
      else if (anchor_position_estimate_en_)
      {
        ROS_WARN("[UWB] No fixed anchors were configured. flag=0 anchors will be estimated from replay ranges before EKF updates start.");
        logEvent(ros::Time::now().toSec(), "WARN",
                 "NO_FIXED_ANCHORS flag0_anchors_will_be_estimated source=replay");
      }
      else
      {
        ROS_WARN("[UWB] No UWB EKF update is possible: all anchors are flag=0 and uwb/anchor_position_estimate_en is false.");
        logEvent(ros::Time::now().toSec(), "WARN",
                 "NO_UWB_EKF_UPDATE no_anchor_positions anchor_position_estimate_en=false");
      }
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
  {
    std::ostringstream oss;
    oss << "SERIAL_STARTED port=" << serial_port_
        << " baud=" << baudrate_
        << " dtr=" << static_cast<int>(dtr_high_)
        << " rts=" << static_cast<int>(rts_high_);
    logEvent(ros::Time::now().toSec(), "INFO", oss.str());
  }

  if (update_en_ && anchors_.empty())
  {
    if (anchor_frame_align_en_)
    {
      ROS_WARN("[UWB] Configured anchors are in an external frame. Anchor frame alignment will transform them before EKF updates start.");
      logEvent(ros::Time::now().toSec(), "WARN",
               "NO_LOCAL_ANCHORS anchor_frame_align source=serial");
    }
    else if (baseline_anchor_init_en_)
    {
      ROS_WARN("[UWB] No fixed anchors were configured. Baseline anchor initialization will use start/end ids and known distance before EKF updates start.");
      logEvent(ros::Time::now().toSec(), "WARN",
               "NO_FIXED_ANCHORS baseline_anchor_init source=serial");
    }
    else if (anchor_position_estimate_en_)
    {
      ROS_WARN("[UWB] No fixed anchors were configured. flag=0 anchors will be estimated from serial ranges before EKF updates start.");
      logEvent(ros::Time::now().toSec(), "WARN",
               "NO_FIXED_ANCHORS flag0_anchors_will_be_estimated source=serial");
    }
    else
    {
      ROS_WARN("[UWB] No UWB EKF update is possible: all anchors are flag=0 and uwb/anchor_position_estimate_en is false.");
      logEvent(ros::Time::now().toSec(), "WARN",
               "NO_UWB_EKF_UPDATE no_anchor_positions anchor_position_estimate_en=false");
    }
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
  nh.param<std::string>("uwb/mode", mode_, "external_anchors");
  nh.param<std::string>("uwb/parser_mode", parser_mode_, "uwb");
  nh.param<std::string>("uwb/log_filename", log_filename_, "uwb_ranges.txt");
  nh.param<int>("uwb/log_flush_stride", log_flush_stride_, 1);
  nh.param<std::string>("uwb/replay_file", replay_file_, "");
  nh.param<std::string>("uwb/replay_time_mode", replay_time_mode_, "relative");
  nh.param<double>("uwb/replay_speed", replay_speed_, 1.0);
  nh.param<double>("uwb/range_scale", range_scale_, 1.0);
  nh.param<double>("uwb/min_range_m", min_range_m_, 0.05);
  nh.param<double>("uwb/max_range_m", max_range_m_, 250.0);
  nh.param<double>("uwb/max_age_s", max_age_s_, 0.5);
  nh.param<int>("uwb/max_queue_size", max_queue_size_, 512);
  nh.param<int>("uwb/min_update_anchors", min_update_anchors_, 1);
  nh.param<double>("uwb/range_noise_m", range_noise_m_, 0.20);
  nh.param<double>("uwb/position_cov_floor_m", position_cov_floor_m_, 0.0);
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

  mode_ = toLower(mode_);
  std::replace(mode_.begin(), mode_.end(), '-', '_');
  const bool legacy_entry_exit_mode = mode_ == "entry_exit_distance" ||
                                      mode_ == "entry_exit" ||
                                      mode_ == "baseline" ||
                                      mode_ == "distance" ||
                                      mode_ == "two_anchor" ||
                                      mode_ == "two_anchors";
  const bool external_anchors_mode = mode_ == "external_anchors" ||
                                     mode_ == "external_anchor" ||
                                     mode_ == "external" ||
                                     mode_ == "anchors" ||
                                     mode_ == "multi_anchor" ||
                                     mode_ == "multi_anchors";
  if (legacy_entry_exit_mode)
  {
    ROS_WARN("[UWB] uwb/mode='%s' is deprecated. Use external_anchors and set anchors 0/1 to [0,0,0] and [0,distance,0].",
             mode_.c_str());
    mode_ = "external_anchors";
  }
  else if (!external_anchors_mode)
  {
    ROS_WARN("[UWB] Unknown uwb/mode='%s'. Falling back to external_anchors.",
             mode_.c_str());
    mode_ = "external_anchors";
  }
  else
  {
    mode_ = "external_anchors";
  }

  int entry_anchor_id = 0;
  int exit_anchor_id = 1;
  nh.param<int>("uwb/entry_anchor_id", entry_anchor_id, 0);
  nh.param<int>("uwb/exit_anchor_id", exit_anchor_id, 1);
  baseline_anchor_start_id_ = entry_anchor_id;
  baseline_anchor_end_id_ = exit_anchor_id;
  anchor_frame_align_start_id_ = entry_anchor_id;
  anchor_frame_align_end_id_ = exit_anchor_id;

  baseline_anchor_init_en_ = false;
  anchor_frame_align_en_ = true;
  anchor_position_estimate_en_ = false;
  nh.param<double>("uwb/entry_exit_distance_m", baseline_distance_m_, 0.0);
  double init_min_motion_m = anchor_frame_align_min_motion_m_;
  nh.param<double>("uwb/init_min_motion_m", init_min_motion_m, init_min_motion_m);
  baseline_init_min_motion_m_ = init_min_motion_m;
  anchor_frame_align_min_motion_m_ = init_min_motion_m;
  bool use_start_range_offset = true;
  nh.param<bool>("uwb/use_start_range_offset", use_start_range_offset, true);
  baseline_use_start_range_offset_ = use_start_range_offset;
  anchor_frame_align_use_start_range_offset_ = use_start_range_offset;

  bool baseline_anchor_init_requested = false;
  nh.param<bool>("uwb/baseline_anchor_init_en", baseline_anchor_init_requested, false);
  if (baseline_anchor_init_requested)
  {
    ROS_WARN("[UWB] uwb/baseline_anchor_init_en is deprecated and ignored. Use external_anchors with anchor positions instead.");
  }
  baseline_anchor_init_en_ = false;
  nh.param<int>("uwb/baseline_anchor_start_id", baseline_anchor_start_id_, baseline_anchor_start_id_);
  nh.param<int>("uwb/baseline_anchor_end_id", baseline_anchor_end_id_, baseline_anchor_end_id_);
  nh.param<double>("uwb/baseline_distance_m", baseline_distance_m_, baseline_distance_m_);
  nh.param<double>("uwb/baseline_init_min_motion_m", baseline_init_min_motion_m_, baseline_init_min_motion_m_);
  nh.param<bool>("uwb/baseline_use_start_range_offset", baseline_use_start_range_offset_, baseline_use_start_range_offset_);
  nh.param<bool>("uwb/anchor_frame_align_en", anchor_frame_align_en_, anchor_frame_align_en_);
  nh.param<int>("uwb/anchor_frame_align_start_id", anchor_frame_align_start_id_, anchor_frame_align_start_id_);
  nh.param<int>("uwb/anchor_frame_align_end_id", anchor_frame_align_end_id_, anchor_frame_align_end_id_);
  nh.param<double>("uwb/anchor_frame_align_min_motion_m", anchor_frame_align_min_motion_m_, anchor_frame_align_min_motion_m_);
  nh.param<bool>("uwb/anchor_frame_align_use_start_range_offset", anchor_frame_align_use_start_range_offset_, anchor_frame_align_use_start_range_offset_);
  nh.param<bool>("uwb/anchor_frame_align_yaw_only", anchor_frame_align_yaw_only_, true);
  nh.param<bool>("uwb/anchor_frame_align_multi_en", anchor_frame_align_multi_en_, true);
  nh.param<int>("uwb/anchor_frame_align_multi_min_anchors", anchor_frame_align_multi_min_anchors_, 3);
  nh.param<int>("uwb/anchor_frame_align_multi_min_samples_per_anchor", anchor_frame_align_multi_min_samples_per_anchor_, 5);
  nh.param<int>("uwb/anchor_frame_align_multi_min_total_samples", anchor_frame_align_multi_min_total_samples_, 30);
  nh.param<int>("uwb/anchor_frame_align_multi_max_samples_per_anchor", anchor_frame_align_multi_max_samples_per_anchor_, 200);
  nh.param<int>("uwb/anchor_frame_align_multi_max_iterations", anchor_frame_align_multi_max_iterations_, 15);
  nh.param<double>("uwb/anchor_frame_align_multi_huber_delta_m", anchor_frame_align_multi_huber_delta_m_, 1.0);
  nh.param<double>("uwb/anchor_frame_align_multi_max_rmse_m", anchor_frame_align_multi_max_rmse_m_, 3.0);
  nh.param<double>("uwb/anchor_frame_align_multi_retry_period_s", anchor_frame_align_multi_retry_period_s_, 1.0);
  nh.param<bool>("uwb/anchor_position_estimate_en", anchor_position_estimate_en_, anchor_position_estimate_en_);
  nh.param<bool>("uwb/start_anchor_origin_en", start_anchor_origin_en_, false);
  nh.param<int>("uwb/start_anchor_origin_id", start_anchor_origin_id_, anchor_frame_align_start_id_);
  nh.param<double>("uwb/start_anchor_origin_tolerance_m", start_anchor_origin_tolerance_m_, 0.20);

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
  position_cov_floor_m_ = std::max(0.0, position_cov_floor_m_);
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
  baseline_distance_m_ = std::max(0.0, baseline_distance_m_);
  baseline_init_min_motion_m_ = std::max(0.1, baseline_init_min_motion_m_);
  anchor_frame_align_min_motion_m_ = std::max(0.1, anchor_frame_align_min_motion_m_);
  anchor_frame_align_multi_min_anchors_ = std::max(3, anchor_frame_align_multi_min_anchors_);
  anchor_frame_align_multi_min_samples_per_anchor_ = std::max(2, anchor_frame_align_multi_min_samples_per_anchor_);
  anchor_frame_align_multi_min_total_samples_ = std::max(
      anchor_frame_align_multi_min_anchors_ * anchor_frame_align_multi_min_samples_per_anchor_,
      anchor_frame_align_multi_min_total_samples_);
  anchor_frame_align_multi_max_samples_per_anchor_ = std::max(
      anchor_frame_align_multi_min_samples_per_anchor_,
      anchor_frame_align_multi_max_samples_per_anchor_);
  anchor_frame_align_multi_max_iterations_ = std::max(1, anchor_frame_align_multi_max_iterations_);
  anchor_frame_align_multi_huber_delta_m_ = std::max(0.0, anchor_frame_align_multi_huber_delta_m_);
  anchor_frame_align_multi_max_rmse_m_ = std::max(0.0, anchor_frame_align_multi_max_rmse_m_);
  anchor_frame_align_multi_retry_period_s_ = std::max(0.0, anchor_frame_align_multi_retry_period_s_);
  start_anchor_origin_tolerance_m_ = std::max(0.0, start_anchor_origin_tolerance_m_);

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
      if (anchor.enabled && !anchor_frame_align_en_) anchors_[anchor.id] = anchor;
    }
  }

  if (legacy_entry_exit_mode && baseline_distance_m_ > 0.0)
  {
    const bool has_start = configured_anchors_.count(anchor_frame_align_start_id_) > 0 &&
                           configured_anchors_[anchor_frame_align_start_id_].enabled;
    const bool has_end = configured_anchors_.count(anchor_frame_align_end_id_) > 0 &&
                         configured_anchors_[anchor_frame_align_end_id_].enabled;
    if (!has_start || !has_end)
    {
      UwbAnchor start_anchor;
      start_anchor.id = anchor_frame_align_start_id_;
      start_anchor.enabled = true;
      start_anchor.estimated = false;
      start_anchor.position_w = V3D::Zero();

      UwbAnchor end_anchor;
      end_anchor.id = anchor_frame_align_end_id_;
      end_anchor.enabled = true;
      end_anchor.estimated = false;
      end_anchor.position_w = V3D(0.0, baseline_distance_m_, 0.0);

      configured_anchors_[start_anchor.id] = start_anchor;
      configured_anchors_[end_anchor.id] = end_anchor;
      if (std::find(anchor_order_.begin(), anchor_order_.end(), start_anchor.id) == anchor_order_.end())
      {
        anchor_order_.push_back(start_anchor.id);
      }
      if (std::find(anchor_order_.begin(), anchor_order_.end(), end_anchor.id) == anchor_order_.end())
      {
        anchor_order_.push_back(end_anchor.id);
      }
      if (!anchor_frame_align_en_)
      {
        anchors_[start_anchor.id] = start_anchor;
        anchors_[end_anchor.id] = end_anchor;
      }
      ROS_WARN("[UWB] Converted deprecated entry/exit distance %.3f m to external anchors: id=%d [0,0,0], id=%d [0,%.3f,0].",
               baseline_distance_m_,
               start_anchor.id,
               end_anchor.id,
               baseline_distance_m_);
    }
  }

  if (start_anchor_origin_en_)
  {
    const auto origin_it = configured_anchors_.find(start_anchor_origin_id_);
    if (origin_it == configured_anchors_.end() || !origin_it->second.enabled)
    {
      ROS_WARN("[UWB] uwb/start_anchor_origin_en is enabled, but origin anchor id=%d is not enabled. Keep normal anchor-frame alignment.",
               start_anchor_origin_id_);
      start_anchor_origin_en_ = false;
    }
    else if (origin_it->second.position_w.norm() > start_anchor_origin_tolerance_m_)
    {
      ROS_WARN("[UWB] uwb/start_anchor_origin_en is enabled, but anchor id=%d position norm %.3f m is larger than tolerance %.3f m. Keep normal anchor-frame alignment.",
               start_anchor_origin_id_,
               origin_it->second.position_w.norm(),
               start_anchor_origin_tolerance_m_);
      start_anchor_origin_en_ = false;
    }
    else
    {
      baseline_anchor_init_en_ = false;
      anchor_frame_align_en_ = false;
      anchor_position_estimate_en_ = false;
      anchor_frame_aligned_ = true;
      anchor_frame_align_R_ext_to_w_.setIdentity();
      anchor_frame_align_t_ext_to_w_.setZero();
      anchors_.clear();
      for (const auto &item : configured_anchors_)
      {
        if (!item.second.enabled) continue;
        UwbAnchor local_anchor = item.second;
        local_anchor.estimated = false;
        anchors_[local_anchor.id] = local_anchor;
      }
      ROS_INFO("[UWB] Start-anchor origin mode enabled: origin id=%d at [0,0,0]. Using %zu configured anchors directly in FAST-LIVO2 local frame.",
               start_anchor_origin_id_, anchors_.size());
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

  ROS_INFO("[UWB] enable=%d update=%d mode=%s parser=%s anchors_for_update=%zu",
           static_cast<int>(en_), static_cast<int>(update_en_),
           mode_.c_str(), parser_mode_.c_str(), anchors_.size());
  if (position_cov_floor_m_ > 0.0)
  {
    ROS_INFO("[UWB] Position covariance floor enabled for UWB updates: %.3f m",
             position_cov_floor_m_);
  }
  if (tag_offset_estimate_en_)
  {
    ROS_WARN("[UWB] Online tag_offset estimation is enabled. Use it only with fixed anchors, enough motion, and checked range outliers.");
  }
  if (anchor_position_estimate_en_)
  {
    ROS_WARN("[UWB] Online anchor position estimation is enabled for flag=0 anchors. Estimates need enough trajectory excitation; tunnel-line motion may not fully constrain 3D.");
  }
  if (baseline_anchor_init_en_)
  {
    ROS_INFO("[UWB] Baseline anchor init enabled: start=%d end=%d distance=%.3f min_motion=%.3f use_start_range_offset=%d",
             baseline_anchor_start_id_,
             baseline_anchor_end_id_,
             configuredBaselineDistance(),
             baseline_init_min_motion_m_,
             static_cast<int>(baseline_use_start_range_offset_));
  }
  if (anchor_frame_align_en_)
  {
    ROS_INFO("[UWB] Anchor frame alignment enabled: start=%d end=%d min_motion=%.3f use_start_range_offset=%d yaw_only=%d multi=%d min_anchors=%d min_total_samples=%d",
             anchor_frame_align_start_id_,
             anchor_frame_align_end_id_,
             anchor_frame_align_min_motion_m_,
             static_cast<int>(anchor_frame_align_use_start_range_offset_),
             static_cast<int>(anchor_frame_align_yaw_only_),
             static_cast<int>(anchor_frame_align_multi_en_),
             anchor_frame_align_multi_min_anchors_,
             anchor_frame_align_multi_min_total_samples_);
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
  replay_file_start_stamp_ = 0.0;
  replay_file_start_stamp_ready_ = false;

  std::string line;
  while (std::getline(replay_file, line))
  {
    line = trimLine(line);
    if (line.empty() || line[0] == '#') continue;
    if (line.find(" UPDATE ") != std::string::npos || line.find("UPDATE used=") != std::string::npos) continue;

    std::istringstream iss(line);
    double stamp = 0.0;
    if (!(iss >> stamp)) continue;

    const size_t first_space = line.find_first_of(" \t");
    const std::string after_stamp = first_space == std::string::npos ? "" : trimLine(line.substr(first_space + 1));
    if (after_stamp.rfind("INFO ", 0) == 0 ||
        after_stamp.rfind("WARN ", 0) == 0 ||
        after_stamp.rfind("ERROR ", 0) == 0)
    {
      continue;
    }
    if (!replay_file_start_stamp_ready_)
    {
      replay_file_start_stamp_ = stamp;
      replay_file_start_stamp_ready_ = true;
    }

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
      raw_line = after_stamp.empty() ? line : after_stamp;
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
    if (!replay_file_start_stamp_ready_)
    {
      replay_file_start_stamp_ = replay_measurements_.front().stamp;
      replay_file_start_stamp_ready_ = true;
    }
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
    if (!replay_file_start_stamp_ready_)
    {
      replay_file_start_stamp_ = replay_measurements_.front().stamp;
      replay_file_start_stamp_ready_ = true;
    }
  }

  while (replay_index_ < replay_measurements_.size())
  {
    const auto &measurement = replay_measurements_[replay_index_];
    double due_time = measurement.stamp;
    if (replay_time_mode_ != "absolute")
    {
      const double effective_replay_speed = ros::Time::isSimTime() ? 1.0 : replay_speed_;
      due_time = replay_ros_start_stamp_ +
                 (measurement.stamp - replay_file_start_stamp_) / effective_replay_speed;
    }

    if (due_time > now) break;
    if (max_age_s_ > 0.0 && now - due_time > max_age_s_)
    {
      ROS_WARN_THROTTLE(3.0,
                        "[UWB] Drop stale replay range: anchor=%d, replay_time=%.6f, now=%.6f, age=%.3f s > max_age_s=%.3f. Check /use_sim_time, rosbag --clock, or set replay_time_mode=relative.",
                        measurement.anchor_id, due_time, now, now - due_time, max_age_s_);
      {
        std::ostringstream oss;
        oss << "DROP_STALE_REPLAY anchor=" << measurement.anchor_id
            << " replay_time=" << due_time
            << " now=" << now
            << " age=" << (now - due_time)
            << " max_age_s=" << max_age_s_
            << " hint=check_use_sim_time_rosbag_clock_or_set_relative";
        logEventThrottled(now, "drop_stale_replay", 3.0, "WARN", oss.str());
      }
      replay_index_++;
      continue;
    }
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
  if (parser_mode_ == "distance") return measurements;

  static const std::regex target_regex(R"(\btarget\s*=\s*([-+]?\d+))", std::regex::icase);
  static const std::regex ok_regex(R"(\bok\s*=\s*([-+]?\d+))", std::regex::icase);
  static const std::regex dist_regex(R"(\bdist\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))", std::regex::icase);
  const bool parse_debug_distance = parser_mode_ == "uwb" || parser_mode_ == "auto";
  if (parse_debug_distance &&
      (line.find("[UWBDBG]") != std::string::npos || line.find("dist=") != std::string::npos))
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

  if (parser_mode_ == "uwb") return measurements;

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

void UwbManager::logEvent(double stamp, const std::string &level, const std::string &message)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!log_file_.is_open()) return;

  log_file_ << std::fixed << std::setprecision(6)
            << stamp << " " << level << " " << message << '\n';
  log_file_.flush();
}

void UwbManager::logEventThrottled(double stamp, const std::string &key, double period_s,
                                   const std::string &level, const std::string &message)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!log_file_.is_open()) return;

  const auto it = event_log_last_stamp_.find(key);
  if (period_s > 0.0 && it != event_log_last_stamp_.end() && stamp - it->second < period_s)
  {
    return;
  }
  event_log_last_stamp_[key] = stamp;

  log_file_ << std::fixed << std::setprecision(6)
            << stamp << " " << level << " " << message << '\n';
  log_file_.flush();
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

void UwbManager::collectAnchorFrameAlignSamples(
    const StatesGroup &state,
    const std::vector<UwbRangeMeasurement> &measurements)
{
  if (!anchor_frame_align_en_ || anchor_frame_aligned_) return;

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;
  for (const auto &measurement : measurements)
  {
    const auto anchor_it = configured_anchors_.find(measurement.anchor_id);
    if (anchor_it == configured_anchors_.end() || !anchor_it->second.enabled) continue;
    if (measurement.range_m <= min_range_m_ || measurement.range_m >= max_range_m_) continue;

    UwbAnchorSample sample;
    sample.tag_position_w = tag_position_w;
    sample.range_m = measurement.range_m;
    sample.stamp = measurement.stamp;
    auto &samples = anchor_frame_align_samples_[measurement.anchor_id];
    samples.push_back(sample);
    while (static_cast<int>(samples.size()) > anchor_frame_align_multi_max_samples_per_anchor_)
    {
      samples.pop_front();
    }
  }
}

bool UwbManager::estimateMultiAnchorFrame(M3D &R_ext_to_w, V3D &t_ext_to_w,
                                          double &rmse, int &used_anchor_count,
                                          int &used_sample_count) const
{
  rmse = std::numeric_limits<double>::infinity();
  used_anchor_count = 0;
  used_sample_count = 0;
  struct AlignSample
  {
    V3D anchor_ext = V3D::Zero();
    V3D tag_w = V3D::Zero();
    double range_m = 0.0;
  };

  std::vector<AlignSample> samples;
  std::map<int, int> samples_per_anchor;
  for (const auto &item : anchor_frame_align_samples_)
  {
    const auto anchor_it = configured_anchors_.find(item.first);
    if (anchor_it == configured_anchors_.end() || !anchor_it->second.enabled) continue;
    if (static_cast<int>(item.second.size()) < anchor_frame_align_multi_min_samples_per_anchor_) continue;

    samples_per_anchor[item.first] = static_cast<int>(item.second.size());
    for (const auto &sample : item.second)
    {
      AlignSample align_sample;
      align_sample.anchor_ext = anchor_it->second.position_w;
      align_sample.tag_w = sample.tag_position_w;
      align_sample.range_m = sample.range_m;
      samples.push_back(align_sample);
    }
  }

  used_anchor_count = static_cast<int>(samples_per_anchor.size());
  used_sample_count = static_cast<int>(samples.size());
  if (used_anchor_count < anchor_frame_align_multi_min_anchors_ ||
      used_sample_count < anchor_frame_align_multi_min_total_samples_)
  {
    return false;
  }

  V3D pivot_ext = samples.front().anchor_ext;
  const auto start_it = configured_anchors_.find(anchor_frame_align_start_id_);
  if (start_it != configured_anchors_.end() && start_it->second.enabled)
  {
    pivot_ext = start_it->second.position_w;
  }
  const V3D pivot_w = R_ext_to_w * pivot_ext + t_ext_to_w;
  const int parameter_dim = anchor_frame_align_yaw_only_ ? 4 : 6;
  const int yaw_seed_count = 12;
  const double huber_delta = anchor_frame_align_multi_huber_delta_m_;
  constexpr double kPi = 3.14159265358979323846;

  double best_score = std::numeric_limits<double>::infinity();
  double best_rmse = std::numeric_limits<double>::infinity();
  M3D best_R = R_ext_to_w;
  V3D best_t = t_ext_to_w;
  const auto robustScore = [&](const M3D &R, const V3D &t) {
    double score = 0.0;
    for (const auto &sample : samples)
    {
      const double error = sample.range_m - (sample.tag_w - (R * sample.anchor_ext + t)).norm();
      const double abs_error = std::fabs(error);
      if (huber_delta > 0.0 && abs_error > huber_delta)
      {
        score += huber_delta * (abs_error - 0.5 * huber_delta);
      }
      else
      {
        score += 0.5 * error * error;
      }
    }
    return score / static_cast<double>(samples.size());
  };

  for (int seed_idx = 0; seed_idx < yaw_seed_count; ++seed_idx)
  {
    const double yaw_offset = 2.0 * kPi * static_cast<double>(seed_idx) /
                              static_cast<double>(yaw_seed_count);
    M3D R = Eigen::AngleAxisd(yaw_offset, V3D::UnitZ()).toRotationMatrix() * R_ext_to_w;
    V3D t = pivot_w - R * pivot_ext;

    if (samples.size() >= 4)
    {
      const V3D ref_center = samples.front().tag_w - R * samples.front().anchor_ext;
      Eigen::MatrixXd A(samples.size() - 1, 3);
      Eigen::VectorXd b(samples.size() - 1);
      for (size_t i = 1; i < samples.size(); ++i)
      {
        const V3D center = samples[i].tag_w - R * samples[i].anchor_ext;
        A.row(static_cast<int>(i - 1)) = 2.0 * (ref_center - center).transpose();
        b(static_cast<int>(i - 1)) = samples[i].range_m * samples[i].range_m -
                                     samples.front().range_m * samples.front().range_m -
                                     center.squaredNorm() + ref_center.squaredNorm();
      }
      Eigen::JacobiSVD<Eigen::MatrixXd> translation_svd(
          A, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const V3D linear_t = translation_svd.solve(b);
      if (linear_t.allFinite() && robustScore(R, linear_t) < robustScore(R, t)) t = linear_t;
    }

    for (int iter = 0; iter < anchor_frame_align_multi_max_iterations_; ++iter)
    {
      Eigen::MatrixXd H(samples.size(), parameter_dim);
      Eigen::VectorXd residual(samples.size());
      for (size_t i = 0; i < samples.size(); ++i)
      {
        const V3D rotated_anchor = R * samples[i].anchor_ext;
        const V3D diff = samples[i].tag_w - (rotated_anchor + t);
        const double predicted = diff.norm();
        if (predicted < 1e-6)
        {
          H.row(static_cast<int>(i)).setZero();
          residual(static_cast<int>(i)) = 0.0;
          continue;
        }
        const V3D direction = diff / predicted;
        residual(static_cast<int>(i)) = samples[i].range_m - predicted;

        const Eigen::Matrix<double, 1, 3> rot_jacobian =
            direction.transpose() * skewSymmetric(rotated_anchor);
        if (anchor_frame_align_yaw_only_)
        {
          H(static_cast<int>(i), 0) = rot_jacobian(2);
          H.block<1, 3>(static_cast<int>(i), 1) = -direction.transpose();
        }
        else
        {
          H.block<1, 3>(static_cast<int>(i), 0) = rot_jacobian;
          H.block<1, 3>(static_cast<int>(i), 3) = -direction.transpose();
        }

        if (huber_delta > 0.0 && std::fabs(residual(static_cast<int>(i))) > huber_delta)
        {
          const double sqrt_weight = std::sqrt(huber_delta / std::fabs(residual(static_cast<int>(i))));
          H.row(static_cast<int>(i)) *= sqrt_weight;
          residual(static_cast<int>(i)) *= sqrt_weight;
        }
      }

      Eigen::MatrixXd normal = H.transpose() * H +
                               Eigen::MatrixXd::Identity(parameter_dim, parameter_dim) * 1e-6;
      Eigen::VectorXd delta = normal.ldlt().solve(H.transpose() * residual);
      if (!delta.allFinite()) break;

      V3D rot_delta = V3D::Zero();
      V3D trans_delta = V3D::Zero();
      if (anchor_frame_align_yaw_only_)
      {
        rot_delta.z() = delta(0);
        trans_delta = delta.segment<3>(1);
      }
      else
      {
        rot_delta = delta.segment<3>(0);
        trans_delta = delta.segment<3>(3);
      }

      const double max_rot_step = 10.0 / 57.29577951308232;
      if (rot_delta.norm() > max_rot_step) rot_delta *= max_rot_step / rot_delta.norm();
      if (trans_delta.norm() > 5.0) trans_delta *= 5.0 / trans_delta.norm();

      if (anchor_frame_align_yaw_only_)
      {
        R = Eigen::AngleAxisd(rot_delta.z(), V3D::UnitZ()).toRotationMatrix() * R;
      }
      else if (rot_delta.norm() > 1e-12)
      {
        R = Eigen::AngleAxisd(rot_delta.norm(), rot_delta.normalized()).toRotationMatrix() * R;
      }
      t += trans_delta;
      if (rot_delta.norm() < 1e-6 && trans_delta.norm() < 1e-4) break;
    }

    const double score = robustScore(R, t);
    std::vector<double> squared_residuals;
    squared_residuals.reserve(samples.size());
    for (const auto &sample : samples)
    {
      const double error = sample.range_m - (sample.tag_w - (R * sample.anchor_ext + t)).norm();
      squared_residuals.push_back(error * error);
    }
    std::sort(squared_residuals.begin(), squared_residuals.end());
    const size_t retained = std::max<size_t>(1, squared_residuals.size() * 8 / 10);
    double trimmed_sum = 0.0;
    for (size_t i = 0; i < retained; ++i) trimmed_sum += squared_residuals[i];
    const double candidate_rmse = std::sqrt(trimmed_sum / static_cast<double>(retained));

    if (score < best_score)
    {
      best_score = score;
      best_rmse = candidate_rmse;
      best_R = R;
      best_t = t;
    }
  }

  if (!std::isfinite(best_rmse)) return false;
  if (anchor_frame_align_multi_max_rmse_m_ > 0.0 &&
      best_rmse > anchor_frame_align_multi_max_rmse_m_)
  {
    rmse = best_rmse;
    return false;
  }

  R_ext_to_w = best_R;
  t_ext_to_w = best_t;
  rmse = best_rmse;
  return true;
}

bool UwbManager::tryAlignAnchorFrame(const StatesGroup &state,
                                     const std::vector<UwbRangeMeasurement> &measurements)
{
  if (!anchor_frame_align_en_) return false;
  if (anchor_frame_aligned_) return true;

  collectAnchorFrameAlignSamples(state, measurements);

  const auto start_it = configured_anchors_.find(anchor_frame_align_start_id_);
  const auto end_it = configured_anchors_.find(anchor_frame_align_end_id_);
  if (start_it == configured_anchors_.end() || end_it == configured_anchors_.end() ||
      !start_it->second.enabled || !end_it->second.enabled)
  {
    logEventThrottled(ros::Time::now().toSec(), "anchor_frame_missing_ids", 3.0, "WARN",
                      "WAIT_ANCHOR_FRAME_ALIGN missing_enabled_start_or_end_anchor hint=set_flag1_and_position_for_align_start_end");
    return false;
  }

  const V3D ext_start = start_it->second.position_w;
  const V3D ext_end = end_it->second.position_w;
  const V3D ext_vec = ext_end - ext_start;
  const double ext_len = ext_vec.norm();
  if (ext_len < 1e-6)
  {
    logEventThrottled(ros::Time::now().toSec(), "anchor_frame_bad_external_baseline", 3.0, "WARN",
                      "WAIT_ANCHOR_FRAME_ALIGN external_start_end_too_close");
    return false;
  }

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;

  if (!anchor_frame_align_start_pose_ready_)
  {
    anchor_frame_align_start_tag_position_w_ = tag_position_w;
    anchor_frame_align_start_pose_ready_ = true;
  }

  for (const auto &measurement : measurements)
  {
    if (!anchor_frame_align_start_range_ready_ &&
        measurement.anchor_id == anchor_frame_align_start_id_ &&
        measurement.range_m > min_range_m_ &&
        measurement.range_m < max_range_m_)
    {
      anchor_frame_align_start_range_m_ = measurement.range_m;
      anchor_frame_align_start_range_ready_ = true;
    }
  }

  const V3D motion = tag_position_w - anchor_frame_align_start_tag_position_w_;
  const double motion_norm = motion.norm();
  if (motion_norm < anchor_frame_align_min_motion_m_)
  {
    std::ostringstream oss;
    oss << "WAIT_ANCHOR_FRAME_ALIGN motion=" << motion_norm
        << " min_motion=" << anchor_frame_align_min_motion_m_
        << " external_baseline=" << ext_len;
    logEventThrottled(ros::Time::now().toSec(), "anchor_frame_wait_motion", 3.0, "WARN", oss.str());
    return false;
  }

  M3D R_ext_to_w = M3D::Identity();
  V3D local_dir = motion / motion_norm;
  if (anchor_frame_align_yaw_only_)
  {
    V3D ext_xy(ext_vec.x(), ext_vec.y(), 0.0);
    V3D local_xy(motion.x(), motion.y(), 0.0);
    const double ext_xy_norm = ext_xy.norm();
    const double local_xy_norm = local_xy.norm();
    if (ext_xy_norm < 1e-6 || local_xy_norm < 1e-6)
    {
      logEventThrottled(ros::Time::now().toSec(), "anchor_frame_bad_yaw_baseline", 3.0, "WARN",
                        "WAIT_ANCHOR_FRAME_ALIGN horizontal_direction_too_small");
      return false;
    }

    local_dir = local_xy / local_xy_norm;
    const double yaw_ext = std::atan2(ext_vec.y(), ext_vec.x());
    const double yaw_local = std::atan2(local_xy.y(), local_xy.x());
    const double yaw = yaw_local - yaw_ext;
    R_ext_to_w = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
  }
  else
  {
    R_ext_to_w = Eigen::Quaterniond::FromTwoVectors(ext_vec / ext_len, local_dir).toRotationMatrix();
  }

  V3D local_start_anchor = anchor_frame_align_start_tag_position_w_;
  if (anchor_frame_align_use_start_range_offset_ && anchor_frame_align_start_range_ready_)
  {
    local_start_anchor -= local_dir * anchor_frame_align_start_range_m_;
  }

  V3D t_ext_to_w = local_start_anchor - R_ext_to_w * ext_start;

  int enabled_anchor_count = 0;
  for (const auto &item : configured_anchors_)
  {
    if (item.second.enabled) enabled_anchor_count++;
  }

  bool used_multi_alignment = false;
  double multi_rmse = 0.0;
  int multi_anchor_count = 0;
  int multi_sample_count = 0;
  if (anchor_frame_align_multi_en_ &&
      enabled_anchor_count >= anchor_frame_align_multi_min_anchors_)
  {
    const double now = ros::Time::now().toSec();
    if (anchor_frame_align_last_multi_attempt_stamp_ >= 0.0 &&
        now - anchor_frame_align_last_multi_attempt_stamp_ < anchor_frame_align_multi_retry_period_s_)
    {
      return false;
    }
    anchor_frame_align_last_multi_attempt_stamp_ = now;
    if (!estimateMultiAnchorFrame(R_ext_to_w, t_ext_to_w, multi_rmse,
                                  multi_anchor_count, multi_sample_count))
    {
      std::ostringstream wait_oss;
      wait_oss << "WAIT_MULTI_ANCHOR_FRAME_ALIGN enabled_anchors=" << enabled_anchor_count
               << " observed_ready_anchors=" << multi_anchor_count
               << " samples=" << multi_sample_count
               << " required_anchors=" << anchor_frame_align_multi_min_anchors_
               << " required_samples=" << anchor_frame_align_multi_min_total_samples_;
      if (std::isfinite(multi_rmse)) wait_oss << " rmse=" << multi_rmse;
      logEventThrottled(ros::Time::now().toSec(), "multi_anchor_frame_wait", 3.0,
                        "WARN", wait_oss.str());
      return false;
    }
    used_multi_alignment = true;
  }

  anchors_.clear();
  anchor_frame_align_R_ext_to_w_ = R_ext_to_w;
  anchor_frame_align_t_ext_to_w_ = t_ext_to_w;
  for (const auto &item : configured_anchors_)
  {
    if (!item.second.enabled) continue;
    UwbAnchor aligned_anchor = item.second;
    aligned_anchor.estimated = true;
    aligned_anchor.position_w = R_ext_to_w * item.second.position_w + t_ext_to_w;
    anchors_[aligned_anchor.id] = aligned_anchor;
  }
  anchor_frame_aligned_ = !anchors_.empty();
  if (!anchor_frame_aligned_)
  {
    logEventThrottled(ros::Time::now().toSec(), "anchor_frame_no_enabled_anchors", 3.0, "WARN",
                      "WAIT_ANCHOR_FRAME_ALIGN no_enabled_anchors_after_transform");
    return false;
  }
  anchor_frame_align_samples_.clear();

  const V3D aligned_start = R_ext_to_w * ext_start + t_ext_to_w;
  const V3D aligned_end = R_ext_to_w * ext_end + t_ext_to_w;
  std::ostringstream oss;
  oss << "ANCHOR_FRAME_ALIGNED start_id=" << anchor_frame_align_start_id_
      << " end_id=" << anchor_frame_align_end_id_
      << " external_baseline=" << ext_len
      << " motion=" << motion_norm
      << " yaw_only=" << static_cast<int>(anchor_frame_align_yaw_only_)
      << " method=" << (used_multi_alignment ? "multi_anchor" : "start_end_pair")
      << " used_anchors=" << (used_multi_alignment ? multi_anchor_count : 2)
      << " used_samples=" << (used_multi_alignment ? multi_sample_count : 0)
      << " rmse=" << (used_multi_alignment ? multi_rmse : 0.0)
      << " start_range=" << (anchor_frame_align_start_range_ready_ ? anchor_frame_align_start_range_m_ : 0.0)
      << " local_start=" << aligned_start.transpose()
      << " local_end=" << aligned_end.transpose()
      << " t_ext_to_w=" << t_ext_to_w.transpose();
  logEvent(ros::Time::now().toSec(), "INFO", oss.str());
  for (const auto &item : anchors_)
  {
    std::ostringstream anchor_oss;
    anchor_oss << "ANCHOR_FRAME_ALIGNED_ANCHOR id=" << item.first
               << " position=" << item.second.position_w.transpose();
    logEvent(ros::Time::now().toSec(), "INFO", anchor_oss.str());
  }

  ROS_INFO("[UWB] Anchor frame aligned (%s): start id=%d [%.3f %.3f %.3f], end id=%d [%.3f %.3f %.3f], external baseline=%.3f m, motion=%.3f m, anchors=%d, samples=%d, rmse=%.3f m.",
           used_multi_alignment ? "multi-anchor" : "start/end pair",
           anchor_frame_align_start_id_,
           aligned_start.x(), aligned_start.y(), aligned_start.z(),
           anchor_frame_align_end_id_,
           aligned_end.x(), aligned_end.y(), aligned_end.z(),
           ext_len,
           motion_norm,
           used_multi_alignment ? multi_anchor_count : 2,
           used_multi_alignment ? multi_sample_count : 0,
           used_multi_alignment ? multi_rmse : 0.0);
  return true;
}

double UwbManager::configuredBaselineDistance() const
{
  if (baseline_distance_m_ > 0.0) return baseline_distance_m_;
  for (const auto &constraint : anchor_distance_constraints_)
  {
    const bool forward = constraint.id_a == baseline_anchor_start_id_ &&
                         constraint.id_b == baseline_anchor_end_id_;
    const bool reverse = constraint.id_a == baseline_anchor_end_id_ &&
                         constraint.id_b == baseline_anchor_start_id_;
    if ((forward || reverse) && constraint.distance_m > 0.0) return constraint.distance_m;
  }
  return 0.0;
}

bool UwbManager::tryInitializeBaselineAnchors(const StatesGroup &state,
                                              const std::vector<UwbRangeMeasurement> &measurements)
{
  if (!baseline_anchor_init_en_) return false;
  if (baseline_anchors_initialized_) return true;

  const double baseline_distance = configuredBaselineDistance();
  if (baseline_distance <= 0.0)
  {
    logEventThrottled(ros::Time::now().toSec(), "baseline_no_distance", 3.0, "WARN",
                      "WAIT_BASELINE_ANCHORS missing_distance hint=set_baseline_distance_m_or_anchor_distance_constraints");
    return false;
  }

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;

  if (!baseline_start_pose_ready_)
  {
    baseline_start_tag_position_w_ = tag_position_w;
    baseline_start_pose_ready_ = true;
  }

  for (const auto &measurement : measurements)
  {
    if (!baseline_start_range_ready_ &&
        measurement.anchor_id == baseline_anchor_start_id_ &&
        measurement.range_m > min_range_m_ &&
        measurement.range_m < max_range_m_)
    {
      baseline_start_range_m_ = measurement.range_m;
      baseline_start_range_ready_ = true;
    }
  }

  const V3D motion = tag_position_w - baseline_start_tag_position_w_;
  const double motion_norm = motion.norm();
  if (motion_norm < baseline_init_min_motion_m_)
  {
    std::ostringstream oss;
    oss << "WAIT_BASELINE_ANCHORS motion=" << motion_norm
        << " min_motion=" << baseline_init_min_motion_m_
        << " distance=" << baseline_distance;
    logEventThrottled(ros::Time::now().toSec(), "baseline_wait_motion", 3.0, "WARN", oss.str());
    return false;
  }

  const V3D baseline_dir = motion / motion_norm;
  V3D start_anchor_pos = baseline_start_tag_position_w_;
  if (baseline_use_start_range_offset_ && baseline_start_range_ready_)
  {
    start_anchor_pos -= baseline_dir * baseline_start_range_m_;
  }
  const V3D end_anchor_pos = start_anchor_pos + baseline_dir * baseline_distance;

  UwbAnchor start_anchor;
  start_anchor.id = baseline_anchor_start_id_;
  start_anchor.enabled = true;
  start_anchor.estimated = true;
  start_anchor.position_w = start_anchor_pos;

  UwbAnchor end_anchor;
  end_anchor.id = baseline_anchor_end_id_;
  end_anchor.enabled = true;
  end_anchor.estimated = true;
  end_anchor.position_w = end_anchor_pos;

  configured_anchors_[start_anchor.id] = start_anchor;
  configured_anchors_[end_anchor.id] = end_anchor;
  anchors_[start_anchor.id] = start_anchor;
  anchors_[end_anchor.id] = end_anchor;
  baseline_anchors_initialized_ = true;

  std::ostringstream oss;
  oss << "BASELINE_ANCHORS_INITIALIZED start_id=" << start_anchor.id
      << " end_id=" << end_anchor.id
      << " distance=" << baseline_distance
      << " motion=" << motion_norm
      << " start_range=" << (baseline_start_range_ready_ ? baseline_start_range_m_ : 0.0)
      << " start_pos=" << start_anchor_pos.transpose()
      << " end_pos=" << end_anchor_pos.transpose();
  logEvent(ros::Time::now().toSec(), "INFO", oss.str());

  ROS_INFO("[UWB] Baseline anchors initialized: start id=%d [%.3f %.3f %.3f], end id=%d [%.3f %.3f %.3f], distance=%.3f m, motion=%.3f m.",
           start_anchor.id,
           start_anchor.position_w.x(), start_anchor.position_w.y(), start_anchor.position_w.z(),
           end_anchor.id,
           end_anchor.position_w.x(), end_anchor.position_w.y(), end_anchor.position_w.z(),
           baseline_distance,
           motion_norm);
  return true;
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
    std::string failure_reason;
    if (!estimateAnchorPosition(anchor_id, estimated_anchor, rmse, rank, &failure_reason))
    {
      if (static_cast<int>(item.second.size()) >= anchor_estimate_min_samples_)
      {
        std::ostringstream oss;
        oss << "ANCHOR_ESTIMATE_PENDING id=" << anchor_id
            << " reason=" << failure_reason
            << " samples=" << item.second.size()
            << " rank=" << rank
            << " rmse=" << rmse;
        logEventThrottled(ros::Time::now().toSec(),
                          "anchor_estimate_pending_" + std::to_string(anchor_id),
                          3.0, "WARN", oss.str());
      }
      continue;
    }

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

bool UwbManager::estimateAnchorPosition(int anchor_id, UwbAnchor &anchor, double &rmse, int &rank,
                                        std::string *failure_reason) const
{
  auto fail = [&](const std::string &reason) {
    if (failure_reason != nullptr) *failure_reason = reason;
    return false;
  };

  const auto samples_it = anchor_samples_.find(anchor_id);
  if (samples_it == anchor_samples_.end()) return fail("no_samples");

  const auto &samples = samples_it->second;
  const int n = static_cast<int>(samples.size());
  if (n < anchor_estimate_min_samples_) return fail("not_enough_samples");

  V3D min_p = samples.front().tag_position_w;
  V3D max_p = samples.front().tag_position_w;
  for (const auto &sample : samples)
  {
    min_p = min_p.cwiseMin(sample.tag_position_w);
    max_p = max_p.cwiseMax(sample.tag_position_w);
  }
  if ((max_p - min_p).norm() < anchor_estimate_min_motion_m_) return fail("not_enough_motion");

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
  if (svd.singularValues().size() == 0) return fail("empty_svd");
  const double max_sv = svd.singularValues()(0);
  rank = 0;
  for (int i = 0; i < svd.singularValues().size(); ++i)
  {
    if (svd.singularValues()(i) > std::max(1e-6, max_sv * 1e-3)) rank++;
  }
  if (rank < anchor_estimate_min_rank_) return fail("rank_too_low");

  V3D estimate = svd.solve(b);
  if (!estimate.allFinite()) return fail("non_finite_linear_solution");

  for (int iter = 0; iter < 8; ++iter)
  {
    Eigen::MatrixXd H(n, 3);
    Eigen::VectorXd residual(n);
    for (int i = 0; i < n; ++i)
    {
      const V3D diff = estimate - samples[i].tag_position_w;
      const double predicted = diff.norm();
      if (predicted < 1e-6) return fail("zero_predicted_range");
      H.row(i) = (diff / predicted).transpose();
      residual(i) = samples[i].range_m - predicted;
    }

    Eigen::Matrix3d normal = H.transpose() * H + Eigen::Matrix3d::Identity() * 1e-6;
    V3D delta = normal.ldlt().solve(H.transpose() * residual);
    if (!delta.allFinite()) return fail("non_finite_gn_step");
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
  if (anchor_estimate_max_rmse_m_ > 0.0 && rmse > anchor_estimate_max_rmse_m_) return fail("rmse_too_high");

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
                           const V3D &trans_add, const V3D &tag_offset_add,
                           const std::string &range_details)
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
            << " ranges_pre_update={" << range_details << "}"
            << '\n';
  log_file_.flush();
}

int UwbManager::applyRangeUpdate(StatesGroup &state)
{
  if (!en_ || !update_en_) return 0;
  const double now = ros::Time::now().toSec();
  if ((anchor_frame_align_en_ && !anchor_frame_align_start_pose_ready_) ||
      (!anchor_frame_align_en_ && baseline_anchor_init_en_ && !baseline_start_pose_ready_))
  {
    const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
    const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;
    if (anchor_frame_align_en_ && !anchor_frame_align_start_pose_ready_)
    {
      anchor_frame_align_start_tag_position_w_ = tag_position_w;
      anchor_frame_align_start_pose_ready_ = true;
    }
    if (!anchor_frame_align_en_ && baseline_anchor_init_en_ && !baseline_start_pose_ready_)
    {
      baseline_start_tag_position_w_ = tag_position_w;
      baseline_start_pose_ready_ = true;
    }
  }

  const std::string source = toLower(input_source_);
  const auto measurements = (source == "file" || source == "txt" || source == "replay") ?
                            takeReplayMeasurements(now) :
                            takeRecentMeasurements(now);
  if (measurements.empty()) return 0;
  const bool anchor_frame_ready = tryAlignAnchorFrame(state, measurements);
  const bool baseline_ready = anchor_frame_align_en_ ? false : tryInitializeBaselineAnchors(state, measurements);
  collectAnchorEstimateSamples(state, measurements);
  if (anchors_.empty())
  {
    if (anchor_frame_align_en_ && !anchor_frame_ready)
    {
      ROS_WARN_THROTTLE(3.0,
                        "[UWB] Waiting for anchor frame alignment before EKF update: start=%d, end=%d, min_motion=%.3f.",
                        anchor_frame_align_start_id_, anchor_frame_align_end_id_,
                        anchor_frame_align_min_motion_m_);
    }
    else if (baseline_anchor_init_en_ && !baseline_ready)
    {
      ROS_WARN_THROTTLE(3.0,
                        "[UWB] Waiting for baseline anchors before EKF update: start=%d, end=%d, distance=%.3f, min_motion=%.3f.",
                        baseline_anchor_start_id_, baseline_anchor_end_id_,
                        configuredBaselineDistance(), baseline_init_min_motion_m_);
    }
    else if (anchor_position_estimate_en_)
    {
      ROS_WARN_THROTTLE(3.0,
                        "[UWB] Waiting for anchor estimates before EKF update: replay/serial measurements=%zu, tracked_anchor_ids=%zu, min_samples=%d.",
                        measurements.size(), anchor_samples_.size(), anchor_estimate_min_samples_);
      {
        std::ostringstream oss;
        oss << "WAIT_ANCHOR_ESTIMATE measurements=" << measurements.size()
            << " tracked_anchor_ids=" << anchor_samples_.size()
            << " min_samples=" << anchor_estimate_min_samples_;
        logEventThrottled(now, "wait_anchor_estimate", 3.0, "WARN", oss.str());
      }
    }
    else
    {
      ROS_WARN_THROTTLE(3.0,
                        "[UWB] Skip EKF update: no anchor positions are available. Set anchor flag=1 with position, or enable uwb/anchor_position_estimate_en.");
      logEventThrottled(now, "skip_no_anchor_positions", 3.0, "WARN",
                        "SKIP_EKF_UPDATE no_anchor_positions hint=set_anchor_flag1_or_enable_anchor_position_estimate");
    }
    return 0;
  }
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
  std::ostringstream range_details;
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
      {
        std::ostringstream oss;
        oss << "REJECT_RANGE anchor=" << measurement.anchor_id
            << " residual=" << residual
            << " max_residual=" << max_residual_m_
            << " measured=" << measurement.range_m
            << " predicted=" << predicted_range
            << " anchor_pos=[" << anchor_it->second.position_w.transpose() << "]"
            << " tag_pos=[" << tag_position_w.transpose() << "]";
        logEventThrottled(ros::Time::now().toSec(),
                          "reject_range_" + std::to_string(measurement.anchor_id),
                          2.0, "WARN", oss.str());
      }
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
    if (row > 0) range_details << ";";
    range_details << "id=" << measurement.anchor_id
                  << ",anchor=[" << anchor_it->second.position_w.transpose() << "]"
                  << ",tag=[" << tag_position_w.transpose() << "]"
                  << ",measured=" << measurement.range_m
                  << ",predicted=" << predicted_range
                  << ",residual=" << residual;
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
      {
        std::ostringstream oss;
        oss << "SKIP_TAG_OFFSET_ESTIMATION used_anchors=" << row
            << " required=" << tag_offset_estimate_min_anchors_
            << " pose_update_still_runs=1";
        logEventThrottled(ros::Time::now().toSec(), "skip_tag_offset_estimation", 3.0, "WARN", oss.str());
      }
    }

    MD(DIM_STATE, DIM_STATE) cov_for_uwb = state.cov;
    if (position_cov_floor_m_ > 0.0)
    {
      const double floor_var = position_cov_floor_m_ * position_cov_floor_m_;
      for (int i = 0; i < 3; ++i)
      {
        const int idx = 3 + i;
        cov_for_uwb(idx, idx) = std::max(cov_for_uwb(idx, idx), floor_var);
      }
    }

    const Eigen::MatrixXd S = H * cov_for_uwb * H.transpose() + R;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success)
    {
      ROS_WARN_THROTTLE(2.0, "[UWB] Skip update: innovation covariance decomposition failed.");
      logEventThrottled(ros::Time::now().toSec(), "skip_cov_decomposition", 2.0, "WARN",
                        "SKIP_EKF_UPDATE innovation_covariance_decomposition_failed");
      return 0;
    }

    const Eigen::MatrixXd K = cov_for_uwb * H.transpose() * ldlt.solve(Eigen::MatrixXd::Identity(row, row));
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
    state.cov = I_KH * cov_for_uwb * I_KH.transpose() + K * R * K.transpose();
    state.cov = 0.5 * (state.cov + state.cov.transpose());
    snapStateForDeterminism(state);

    logUpdate(ros::Time::now().toSec(), row, z.norm(), rot_add, trans_add,
              V3D::Zero(), range_details.str());
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
  if (position_cov_floor_m_ > 0.0)
  {
    const double floor_var = position_cov_floor_m_ * position_cov_floor_m_;
    for (int i = 0; i < 3; ++i)
    {
      const int idx = 3 + i;
      P_joint(idx, idx) = std::max(P_joint(idx, idx), floor_var);
    }
  }
  const double tag_process_var = tag_offset_process_noise_m_ * tag_offset_process_noise_m_;
  P_joint.block(DIM_STATE, DIM_STATE, 3, 3) =
      tag_offset_cov_ + M3D::Identity() * tag_process_var;

  const Eigen::MatrixXd S = H_joint * P_joint * H_joint.transpose() + R;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success)
  {
    ROS_WARN_THROTTLE(2.0, "[UWB] Skip update: innovation covariance decomposition failed.");
    logEventThrottled(ros::Time::now().toSec(), "skip_cov_decomposition", 2.0, "WARN",
                      "SKIP_EKF_UPDATE innovation_covariance_decomposition_failed");
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

  logUpdate(ros::Time::now().toSec(), row, z.norm(), rot_add, trans_add,
            tag_offset_add, range_details.str());
  ROS_INFO_THROTTLE(1.0, "[UWB] EKF update used=%d residual_norm=%.3f trans_add=%.4f m tag_offset=[%.3f %.3f %.3f]",
                    row, z.norm(), trans_add.norm(),
                    tag_offset_est_body_.x(), tag_offset_est_body_.y(), tag_offset_est_body_.z());
  return row;
}
