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
#include <set>
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

const char *uwbStateName(int state)
{
  switch (state)
  {
    case 0: return "NORMAL";
    case 1: return "SUSPECT";
    case 2: return "LOST";
    default: return "UNKNOWN";
  }
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

  const std::string source = toLower(input_source_);
  if (source == "file" || source == "txt" || source == "replay")
  {
    if (!loadReplayFile())
    {
      ROS_WARN("[UWB] Replay source is enabled, but replay file could not be loaded.");
      logEvent(ros::Time::now().toSec(), "WARN", "REPLAY_LOAD_FAILED file=" + replay_file_);
      return false;
    }
    ROS_INFO("[UWB] Replay source loaded: file=%s measurements=%zu time_mode=sensor_relative replay_speed=1.000(ignored) start_offset=%.3f s",
             replay_file_.c_str(), replay_measurements_.size(), replay_start_offset_s_);
    {
      std::ostringstream oss;
      oss << "REPLAY_LOADED file=" << replay_file_
          << " measurements=" << replay_measurements_.size()
          << " file_start_stamp=" << replay_file_start_stamp_
          << " first_measurement_stamp=" << (replay_measurements_.empty() ? 0.0 : replay_measurements_.front().stamp)
          << " time_mode=sensor_relative"
          << " replay_speed=1.000(ignored)"
          << " start_offset=" << replay_start_offset_s_
          << " replay_start_lidar_stamp=pending_until_first_slam_timestamp"
          << " start_offset_applied=" << static_cast<int>(std::fabs(replay_start_offset_s_) > 1e-12)
          << " offset_semantics=positive_delay_negative_skip";
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
  nh.param<bool>("uwb/enable", en_, en_);
  nh.param<bool>("uwb/update_en", update_en_, true);
  nh.param<bool>("uwb/update_enable", update_en_, update_en_);
  nh.param<bool>("uwb/residual_debug_only", residual_debug_only_, false);
  nh.param<bool>("uwb/update_xy_only", update_xy_only_, true);
  nh.param<bool>("uwb/use_3d_range_model", use_3d_range_model_, true);
  nh.param<bool>("uwb/update_z", update_z_, false);
  nh.param<bool>("uwb/update_orientation", update_orientation_, false);
  nh.param<std::string>("uwb/source", input_source_, "serial");
  nh.param<std::string>("uwb/serial_port", serial_port_, "/dev/ttyUSB0");
  nh.param<int>("uwb/baudrate", baudrate_, 115200);
  nh.param<bool>("uwb/dtr", dtr_high_, true);
  nh.param<bool>("uwb/rts", rts_high_, false);
  nh.param<std::string>("uwb/mode", mode_, "entry_exit_distance");
  nh.param<std::string>("uwb/parser_mode", parser_mode_, "uwb");
  nh.param<std::string>("uwb/log_filename", log_filename_, "uwb_ranges.txt");
  nh.param<int>("uwb/log_flush_stride", log_flush_stride_, 1);
  nh.param<std::string>("uwb/replay_file", replay_file_, "");
  std::string replay_start_offset_param_path = "default";
  bool replay_start_offset_param_found = false;
  double replay_start_offset_param = 0.0;
  double replay_start_offset_primary = 0.0;
  double replay_start_offset_alias = 0.0;
  double replay_start_offset_short_alias = 0.0;
  const bool has_replay_start_offset_s =
      nh.getParam("uwb/replay_start_offset_s", replay_start_offset_primary);
  const bool has_replay_start_offset =
      nh.getParam("uwb/replay_start_offset", replay_start_offset_alias);
  const bool has_start_offset =
      nh.getParam("uwb/start_offset", replay_start_offset_short_alias);
  if (has_replay_start_offset_s)
  {
    replay_start_offset_param = replay_start_offset_primary;
    replay_start_offset_param_path = "uwb/replay_start_offset_s";
    replay_start_offset_param_found = true;
  }
  if (has_replay_start_offset &&
      (!replay_start_offset_param_found ||
       (std::fabs(replay_start_offset_param) < 1e-12 && std::fabs(replay_start_offset_alias) > 1e-12)))
  {
    replay_start_offset_param = replay_start_offset_alias;
    replay_start_offset_param_path = "uwb/replay_start_offset";
    replay_start_offset_param_found = true;
  }
  if (has_start_offset &&
      (!replay_start_offset_param_found ||
       (std::fabs(replay_start_offset_param) < 1e-12 && std::fabs(replay_start_offset_short_alias) > 1e-12)))
  {
    replay_start_offset_param = replay_start_offset_short_alias;
    replay_start_offset_param_path = "uwb/start_offset";
    replay_start_offset_param_found = true;
  }
  if ((has_replay_start_offset_s && has_replay_start_offset &&
       std::fabs(replay_start_offset_primary - replay_start_offset_alias) > 1e-9) ||
      (has_replay_start_offset_s && has_start_offset &&
       std::fabs(replay_start_offset_primary - replay_start_offset_short_alias) > 1e-9) ||
      (has_replay_start_offset && has_start_offset &&
       std::fabs(replay_start_offset_alias - replay_start_offset_short_alias) > 1e-9))
  {
    ROS_WARN("[UWB] Conflicting replay start offset params: replay_start_offset_s=%.6f(%d), replay_start_offset=%.6f(%d), start_offset=%.6f(%d). Use %s=%.6f.",
             replay_start_offset_primary, static_cast<int>(has_replay_start_offset_s),
             replay_start_offset_alias, static_cast<int>(has_replay_start_offset),
             replay_start_offset_short_alias, static_cast<int>(has_start_offset),
             replay_start_offset_param_path.c_str(), replay_start_offset_param);
  }
  replay_start_offset_s_ = replay_start_offset_param_found ? replay_start_offset_param : 0.0;
  nh.param<double>("uwb/range_scale", range_scale_, 1.0);
  nh.param<double>("uwb/min_range_m", min_range_m_, 0.05);
  nh.param<double>("uwb/max_range_m", max_range_m_, 250.0);
  nh.param<double>("uwb/max_age_s", max_age_s_, 0.5);
  double replay_match_threshold_param = -1.0;
  nh.param<double>("uwb/replay_match_threshold_s", replay_match_threshold_param, -1.0);
  replay_match_threshold_s_ = replay_match_threshold_param >= 0.0 ? replay_match_threshold_param : max_age_s_;
  nh.param<int>("uwb/max_queue_size", max_queue_size_, 512);
  nh.param<int>("uwb/min_anchors", min_update_anchors_, 2);
  nh.param<int>("uwb/min_update_anchors", min_update_anchors_, min_update_anchors_);
  nh.param<int>("uwb/min_anchors_for_update", min_anchors_for_update_, 3);
  nh.param<int>("uwb/prefer_anchors", prefer_anchors_, 3);
  nh.param<double>("uwb/sigma", range_noise_m_, 0.10);
  nh.param<double>("uwb/range_noise_m", range_noise_m_, range_noise_m_);
  nh.param<double>("uwb/position_cov_floor_m", position_cov_floor_m_, 0.0);
  nh.param<double>("uwb/max_residual_m", max_residual_m_, 3.0);
  nh.param<double>("uwb/max_residual_rms", max_residual_rms_, 0.50);
  nh.param<double>("uwb/max_xy_correction_normal", max_xy_correction_normal_, 0.50);
  nh.param<double>("uwb/max_update_step_xy", max_update_step_xy_, 0.10);
  nh.param<double>("uwb/two_anchor_sigma_scale", two_anchor_sigma_scale_, 5.0);
  nh.param<int>("uwb/require_consecutive_good_updates", require_consecutive_good_updates_, 3);
  nh.param<double>("uwb/good_residual_rms", good_residual_rms_, 0.30);
  nh.param<bool>("uwb/suspect_hold_en", suspect_hold_en_, false);
  nh.param<bool>("uwb/lost_hold_en", lost_hold_en_, false);
  nh.param<double>("uwb/large_correction_warn_threshold", large_correction_warn_threshold_, 0.50);
  nh.param<double>("uwb/large_correction_reject_threshold", large_correction_reject_threshold_, 3.0);
  nh.param<std::string>("uwb/anchor_file", anchor_file_, "");
  nh.param<bool>("uwb/stale_repeat_filter_en", stale_repeat_filter_en_, true);
  nh.param<double>("uwb/stale_repeat_epsilon_m", stale_repeat_epsilon_m_, 0.001);
  nh.param<int>("uwb/stale_repeat_max_count", stale_repeat_max_count_, 3);
  nh.param<double>("uwb/stale_repeat_max_duration_s", stale_repeat_max_duration_s_, 2.0);
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
  const bool entry_exit_mode = mode_ == "entry_exit_distance" ||
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
  if (!entry_exit_mode && !external_anchors_mode)
  {
    ROS_WARN("[UWB] Unknown uwb/mode='%s'. Use entry_exit_distance or external_anchors. Falling back to entry_exit_distance.",
             mode_.c_str());
    mode_ = "entry_exit_distance";
  }
  else
  {
    mode_ = external_anchors_mode ? "external_anchors" : "entry_exit_distance";
  }

  int entry_anchor_id = 0;
  int exit_anchor_id = 1;
  nh.param<int>("uwb/entry_anchor_id", entry_anchor_id, 0);
  nh.param<int>("uwb/exit_anchor_id", exit_anchor_id, 1);
  baseline_anchor_start_id_ = entry_anchor_id;
  baseline_anchor_end_id_ = exit_anchor_id;
  anchor_frame_align_start_id_ = entry_anchor_id;
  anchor_frame_align_end_id_ = exit_anchor_id;

  baseline_anchor_init_en_ = (mode_ != "external_anchors" && !external_anchors_mode);
  anchor_frame_align_en_ = external_anchors_mode;
  anchor_position_estimate_en_ = false;
  nh.param<double>("uwb/entry_exit_distance_m", baseline_distance_m_, 0.0);
  double init_min_motion_m = external_anchors_mode ? anchor_frame_align_min_motion_m_ : baseline_init_min_motion_m_;
  nh.param<double>("uwb/init_min_motion_m", init_min_motion_m, init_min_motion_m);
  baseline_init_min_motion_m_ = init_min_motion_m;
  anchor_frame_align_min_motion_m_ = init_min_motion_m;
  bool use_start_range_offset = true;
  nh.param<bool>("uwb/use_start_range_offset", use_start_range_offset, true);
  baseline_use_start_range_offset_ = use_start_range_offset;
  anchor_frame_align_use_start_range_offset_ = use_start_range_offset;

  nh.param<bool>("uwb/baseline_anchor_init_en", baseline_anchor_init_en_, baseline_anchor_init_en_);
  nh.param<int>("uwb/baseline_anchor_start_id", baseline_anchor_start_id_, baseline_anchor_start_id_);
  nh.param<int>("uwb/baseline_anchor_end_id", baseline_anchor_end_id_, baseline_anchor_end_id_);
  nh.param<double>("uwb/baseline_distance_m", baseline_distance_m_, baseline_distance_m_);
  nh.param<double>("uwb/baseline_init_min_motion_m", baseline_init_min_motion_m_, baseline_init_min_motion_m_);
  nh.param<bool>("uwb/baseline_use_start_range_offset", baseline_use_start_range_offset_, baseline_use_start_range_offset_);
  nh.param<bool>("uwb/anchor_frame_align_en", anchor_frame_align_en_, anchor_frame_align_en_);
  bool anchors_in_livo_frame_deprecated = false;
  if (nh.getParam("uwb/anchors_in_livo_frame", anchors_in_livo_frame_deprecated) &&
      anchors_in_livo_frame_deprecated)
  {
    ROS_WARN("[UWB] uwb/anchors_in_livo_frame is deprecated. Use anchor_frame_align_en=false instead.");
    anchor_frame_align_en_ = false;
  }
  bool skip_anchor_frame_align = false;
  nh.param<bool>("uwb/skip_anchor_frame_align", skip_anchor_frame_align, false);
  if (skip_anchor_frame_align)
  {
    ROS_WARN("[UWB] uwb/skip_anchor_frame_align is deprecated. Use anchor_frame_align_en=false instead.");
    anchor_frame_align_en_ = false;
  }
  nh.param<int>("uwb/anchor_frame_align_start_id", anchor_frame_align_start_id_, anchor_frame_align_start_id_);
  nh.param<int>("uwb/anchor_frame_align_end_id", anchor_frame_align_end_id_, anchor_frame_align_end_id_);
  nh.param<double>("uwb/anchor_frame_align_min_motion_m", anchor_frame_align_min_motion_m_, anchor_frame_align_min_motion_m_);
  nh.param<double>("uwb/anchor_frame_align_min_duration_s", anchor_frame_align_min_duration_s_, 30.0);
  nh.param<int>("uwb/anchor_frame_align_min_ranges", anchor_frame_align_min_ranges_, 30);
  nh.param<int>("uwb/anchor_frame_align_min_anchors", anchor_frame_align_min_anchors_, 3);
  nh.param<double>("uwb/anchor_frame_align_success_rms_m", anchor_frame_align_success_rms_m_, 0.50);
  nh.param<double>("uwb/anchor_frame_align_success_max_residual_m", anchor_frame_align_success_max_residual_m_, 1.50);
  nh.param<double>("uwb/anchor_frame_align_validation_duration_s", anchor_frame_align_validation_duration_s_, 5.0);
  nh.param<bool>("uwb/anchor_frame_align_use_start_range_offset", anchor_frame_align_use_start_range_offset_, anchor_frame_align_use_start_range_offset_);
  nh.param<bool>("uwb/anchor_frame_align_yaw_only", anchor_frame_align_yaw_only_, true);
  nh.param<bool>("uwb/anchor_position_estimate_en", anchor_position_estimate_en_, anchor_position_estimate_en_);
  if (!anchor_frame_align_en_) anchor_frame_aligned_ = true;

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
  input_source_ = toLower(input_source_);
  ROS_INFO("[UWB] replay timing param primary=%s alias1=%s alias2=%s found=%d used=%s start_offset=%.3f match_threshold=%.3f time_mode=sensor_relative replay_speed=1.000(ignored) semantics=positive_delay_negative_skip",
           nh.resolveName("uwb/replay_start_offset_s").c_str(),
           nh.resolveName("uwb/replay_start_offset").c_str(),
           nh.resolveName("uwb/start_offset").c_str(),
           static_cast<int>(replay_start_offset_param_found),
           replay_start_offset_param_found ? nh.resolveName(replay_start_offset_param_path).c_str() : "default",
           replay_start_offset_s_, replay_match_threshold_s_);
  if ((input_source_ == "file" || input_source_ == "txt" || input_source_ == "replay") &&
      !replay_start_offset_param_found && replay_start_offset_s_ == 0.0)
  {
    ROS_WARN("[UWB] replay start_offset remains default 0. If you expected a delayed replay, check YAML parameter path: %s",
             nh.resolveName("uwb/replay_start_offset_s").c_str());
  }
  min_range_m_ = std::max(0.0, min_range_m_);
  max_range_m_ = std::max(min_range_m_, max_range_m_);
  max_age_s_ = std::max(0.0, max_age_s_);
  replay_match_threshold_s_ = std::max(0.0, replay_match_threshold_s_);
  max_queue_size_ = std::max(8, max_queue_size_);
  min_update_anchors_ = std::max(2, min_update_anchors_);
  min_anchors_for_update_ = std::max(min_update_anchors_, min_anchors_for_update_);
  prefer_anchors_ = std::max(min_update_anchors_, prefer_anchors_);
  range_noise_m_ = std::max(1e-3, range_noise_m_);
  max_residual_rms_ = std::max(0.0, max_residual_rms_);
  max_xy_correction_normal_ = std::max(0.0, max_xy_correction_normal_);
  max_update_step_xy_ = std::max(0.0, max_update_step_xy_);
  two_anchor_sigma_scale_ = std::max(1.0, two_anchor_sigma_scale_);
  require_consecutive_good_updates_ = std::max(0, require_consecutive_good_updates_);
  good_residual_rms_ = std::max(0.0, good_residual_rms_);
  large_correction_warn_threshold_ = std::max(max_xy_correction_normal_, large_correction_warn_threshold_);
  large_correction_reject_threshold_ = std::max(large_correction_warn_threshold_, large_correction_reject_threshold_);
  position_cov_floor_m_ = std::max(0.0, position_cov_floor_m_);
  max_residual_m_ = std::max(0.0, max_residual_m_);
  stale_repeat_epsilon_m_ = std::max(0.0, stale_repeat_epsilon_m_);
  stale_repeat_max_count_ = std::max(1, stale_repeat_max_count_);
  stale_repeat_max_duration_s_ = std::max(0.0, stale_repeat_max_duration_s_);
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
  anchor_frame_align_min_duration_s_ = std::max(0.0, anchor_frame_align_min_duration_s_);
  anchor_frame_align_min_ranges_ = std::max(1, anchor_frame_align_min_ranges_);
  anchor_frame_align_min_anchors_ = std::max(3, anchor_frame_align_min_anchors_);
  anchor_frame_align_success_rms_m_ = std::max(0.0, anchor_frame_align_success_rms_m_);
  anchor_frame_align_success_max_residual_m_ = std::max(0.0, anchor_frame_align_success_max_residual_m_);
  anchor_frame_align_validation_duration_s_ = std::max(0.0, anchor_frame_align_validation_duration_s_);

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

  std::vector<int> enabled_configured_anchor_ids;
  for (const auto &id : anchor_order_)
  {
    const auto anchor_it = configured_anchors_.find(id);
    if (anchor_it != configured_anchors_.end() && anchor_it->second.enabled)
    {
      enabled_configured_anchor_ids.push_back(id);
    }
  }

  if (anchor_frame_align_en_ && enabled_configured_anchor_ids.size() == 2)
  {
    // ponytail: two anchors cannot constrain a free yaw+translation range fit; use the known baseline and walking direction instead.
    two_anchor_baseline_mode_ = true;
    anchor_frame_align_en_ = false;
    anchor_frame_aligned_ = false;
    baseline_anchor_init_en_ = true;
    baseline_anchors_initialized_ = false;
    anchors_.clear();

    const bool entry_exit_ids_available =
        std::find(enabled_configured_anchor_ids.begin(), enabled_configured_anchor_ids.end(), entry_anchor_id) !=
            enabled_configured_anchor_ids.end() &&
        std::find(enabled_configured_anchor_ids.begin(), enabled_configured_anchor_ids.end(), exit_anchor_id) !=
            enabled_configured_anchor_ids.end();
    baseline_anchor_start_id_ = entry_exit_ids_available ? entry_anchor_id : enabled_configured_anchor_ids[0];
    baseline_anchor_end_id_ = entry_exit_ids_available ? exit_anchor_id : enabled_configured_anchor_ids[1];

    const auto start_anchor_it = configured_anchors_.find(baseline_anchor_start_id_);
    const auto end_anchor_it = configured_anchors_.find(baseline_anchor_end_id_);
    if (start_anchor_it != configured_anchors_.end() && end_anchor_it != configured_anchors_.end())
    {
      const double baseline_from_positions =
          (end_anchor_it->second.position_w - start_anchor_it->second.position_w).norm();
      if (baseline_from_positions > 1e-6) baseline_distance_m_ = baseline_from_positions;
    }
    baseline_use_start_range_offset_ = false;
    min_anchors_for_update_ = std::min(min_anchors_for_update_, 2);
    prefer_anchors_ = std::max(2, std::min(prefer_anchors_, 2));

    ROS_INFO("[UWB] Two enabled anchors detected with anchor_frame_align_en=true. Use baseline mode: start=%d end=%d distance=%.3f min_motion=%.3f.",
             baseline_anchor_start_id_, baseline_anchor_end_id_, configuredBaselineDistance(), baseline_init_min_motion_m_);
    logEvent(ros::Time::now().toSec(), "INFO",
             "TWO_ANCHOR_BASELINE_MODE start_id=" + std::to_string(baseline_anchor_start_id_) +
                 " end_id=" + std::to_string(baseline_anchor_end_id_) +
                 " distance=" + std::to_string(configuredBaselineDistance()) +
                 " use_start_range_offset=0");
  }

  ROS_INFO("[UWB] enable=%d update=%d mode=%s parser=%s anchors_for_update=%zu",
           static_cast<int>(en_), static_cast<int>(update_en_),
           mode_.c_str(), parser_mode_.c_str(), anchors_.size());
  const std::string coordinate_mode_log =
      anchor_frame_align_en_ ? "external_uwb_frame" :
      (baseline_anchor_init_en_ ? "baseline_from_motion" : "fast_livo_world");
  const std::string align_method_log =
      anchor_frame_align_en_ ? "range_yaw_tx_ty_validation" :
      (baseline_anchor_init_en_ ? (two_anchor_baseline_mode_ ? "two_anchor_baseline" : "entry_exit_baseline") :
                                  "manual_yaml_direct");
  ROS_INFO("[UWB] anchor_frame_align_en=%d coordinate_mode=%s align_method=%s",
           static_cast<int>(anchor_frame_align_en_),
           coordinate_mode_log.c_str(),
           align_method_log.c_str());
  ROS_INFO("[UWB] range_model=%s update_xy_only=%d update_z=%d update_orientation=%d min_anchors=%d prefer_anchors=%d sigma=%.3f",
           use_3d_range_model_ ? "3d" : "legacy_xy",
           static_cast<int>(update_xy_only_),
           static_cast<int>(update_z_),
           static_cast<int>(update_orientation_),
           min_update_anchors_,
           prefer_anchors_,
           range_noise_m_);
  ROS_INFO("[UWB] update_strategy residual_debug_only=%d max_update_step_xy=%.3f min_anchors_for_update=%d two_anchor_sigma_scale=%.3f require_good=%d good_rms=%.3f suspect_hold=%d lost_hold=%d",
           static_cast<int>(residual_debug_only_),
           max_update_step_xy_,
           min_anchors_for_update_,
           two_anchor_sigma_scale_,
           require_consecutive_good_updates_,
           good_residual_rms_,
           static_cast<int>(suspect_hold_en_),
           static_cast<int>(lost_hold_en_));
  if (!use_3d_range_model_)
  {
    ROS_WARN("[UWB] uwb/use_3d_range_model=false uses the old horizontal-distance model. For real UWB ranges keep it true.");
  }
  if (update_z_ || update_orientation_)
  {
    ROS_WARN("[UWB] UWB z/orientation update is enabled. Default project policy is xy-only; use only after calibration.");
  }
  if (stale_repeat_filter_en_)
  {
    ROS_INFO("[UWB] Stale repeat filter enabled: epsilon=%.4f m max_count=%d max_duration=%.3f s",
             stale_repeat_epsilon_m_, stale_repeat_max_count_, stale_repeat_max_duration_s_);
  }
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
    ROS_INFO("[UWB] Anchor frame alignment enabled: min_duration=%.3f min_ranges=%d min_anchors=%d min_motion=%.3f validation_duration=%.3f",
             anchor_frame_align_min_duration_s_,
             anchor_frame_align_min_ranges_,
             anchor_frame_align_min_anchors_,
             anchor_frame_align_min_motion_m_,
             anchor_frame_align_validation_duration_s_);
  }
  if (!anchor_frame_align_en_ && !baseline_anchor_init_en_)
  {
    ROS_INFO("[UWB] Anchor positions are used directly in FAST-LIVO frame; anchor frame alignment is skipped.");
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
  repeated_range_states_.clear();

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

    auto measurements = filterRepeatedRanges(parseLine(raw_line, stamp), "replay");
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

std::vector<UwbRangeMeasurement> UwbManager::takeReplayMeasurements(double current_lidar_stamp, double lidar_start_stamp)
{
  std::vector<UwbRangeMeasurement> measurements;
  if (replay_measurements_.empty() || replay_index_ >= replay_measurements_.size()) return measurements;
  if (!std::isfinite(current_lidar_stamp) || !std::isfinite(lidar_start_stamp) ||
      current_lidar_stamp <= 0.0 || lidar_start_stamp <= 0.0)
  {
    return measurements;
  }

  const double slam_relative_time = current_lidar_stamp - lidar_start_stamp;
  const double match_threshold = std::max(0.0, replay_match_threshold_s_);
  if (slam_relative_time < 0.0) return measurements;

  if (!replay_started_)
  {
    replay_started_ = true;
    if (!replay_file_start_stamp_ready_)
    {
      replay_file_start_stamp_ = replay_measurements_.front().stamp;
      replay_file_start_stamp_ready_ = true;
    }
    const double first_measurement_stamp = replay_measurements_.empty() ? 0.0 : replay_measurements_.front().stamp;
    std::ostringstream oss;
    oss << "REPLAY_TIMING_START time_mode=sensor_relative"
        << " file_start_stamp=" << replay_file_start_stamp_
        << " first_measurement_stamp=" << first_measurement_stamp
        << " replay_speed=1.000(ignored)"
        << " start_offset=" << replay_start_offset_s_
        << " lidar_start_stamp=" << lidar_start_stamp
        << " current_lidar_stamp=" << current_lidar_stamp
        << " slam_relative_time=" << slam_relative_time
        << " match_threshold=" << match_threshold
        << " first_relative_replay_time=" << (replay_start_offset_s_ + first_measurement_stamp - replay_file_start_stamp_)
        << " start_offset_applied=" << static_cast<int>(std::fabs(replay_start_offset_s_) > 1e-12)
        << " offset_semantics=positive_delay_negative_skip";
    logEvent(current_lidar_stamp, "INFO", oss.str());
    ROS_INFO("[UWB] Replay timing starts: time_mode=sensor_relative file_start_stamp=%.6f first_measurement_stamp=%.6f replay_speed=1.000(ignored) start_offset=%.3f lidar_start_stamp=%.6f current_lidar_stamp=%.6f slam_relative=%.3f match_threshold=%.3f applied=%d semantics=positive_delay_negative_skip",
             replay_file_start_stamp_, first_measurement_stamp, replay_start_offset_s_,
             lidar_start_stamp, current_lidar_stamp, slam_relative_time, match_threshold,
             static_cast<int>(std::fabs(replay_start_offset_s_) > 1e-12));
  }

  if (replay_last_slam_relative_time_ >= 0.0 &&
      slam_relative_time + match_threshold < replay_last_slam_relative_time_)
  {
    ROS_WARN("[UWB] Replay SLAM time moved backward: last=%.6f current=%.6f. Reset replay index.",
             replay_last_slam_relative_time_, slam_relative_time);
    std::ostringstream oss;
    oss << "REPLAY_SLAM_TIME_RESET last_slam_relative=" << replay_last_slam_relative_time_
        << " current_slam_relative=" << slam_relative_time
        << " match_threshold=" << match_threshold;
    logEvent(current_lidar_stamp, "WARN", oss.str());
    replay_index_ = 0;
  }
  replay_last_slam_relative_time_ = slam_relative_time;

  if (replay_start_offset_s_ > 0.0 && slam_relative_time + match_threshold < replay_start_offset_s_)
  {
    ROS_INFO_THROTTLE(3.0,
                      "[UWB] UWB_REPLAY_WAIT_START_OFFSET slam_relative=%.3f start_offset=%.3f threshold=%.3f",
                      slam_relative_time, replay_start_offset_s_, match_threshold);
    std::ostringstream oss;
    oss << "UWB_REPLAY_WAIT_START_OFFSET slam_relative=" << slam_relative_time
        << " start_offset=" << replay_start_offset_s_
        << " match_threshold=" << match_threshold
        << " lidar_start_stamp=" << lidar_start_stamp
        << " current_lidar_stamp=" << current_lidar_stamp
        << " file_start_stamp=" << replay_file_start_stamp_
        << " first_measurement_stamp=" << replay_measurements_.front().stamp
        << " start_offset_applied=1";
    logEventThrottled(current_lidar_stamp, "uwb_replay_wait_start_offset", 3.0, "INFO", oss.str());
    return measurements;
  }

  while (replay_index_ < replay_measurements_.size())
  {
    const auto &measurement = replay_measurements_[replay_index_];
    const double uwb_relative_time = replay_start_offset_s_ + (measurement.stamp - replay_file_start_stamp_);
    if (uwb_relative_time < -1e-9)
    {
      std::ostringstream oss;
      oss << "SKIP_REPLAY_BEFORE_ZERO anchor=" << measurement.anchor_id
          << " measurement_stamp=" << measurement.stamp
          << " file_start_stamp=" << replay_file_start_stamp_
          << " measurement_relative=" << (measurement.stamp - replay_file_start_stamp_)
          << " start_offset=" << replay_start_offset_s_
          << " uwb_relative_time=" << uwb_relative_time;
      logEventThrottled(current_lidar_stamp, "skip_replay_before_zero", 3.0, "INFO", oss.str());
      replay_index_++;
      continue;
    }

    const double dt = uwb_relative_time - slam_relative_time;
    if (dt > match_threshold) break;
    if (dt < -match_threshold)
    {
      ROS_WARN_THROTTLE(3.0,
                        "[UWB] Drop stale replay range: anchor=%d, slam_relative=%.6f, uwb_relative=%.6f, dt=%.3f s > threshold=%.3f.",
                        measurement.anchor_id, slam_relative_time, uwb_relative_time, -dt, match_threshold);
      {
        std::ostringstream oss;
        oss << "DROP_STALE_REPLAY anchor=" << measurement.anchor_id
            << " slam_relative_time=" << slam_relative_time
            << " uwb_relative_time=" << uwb_relative_time
            << " dt=" << dt
            << " abs_dt=" << std::fabs(dt)
            << " match_threshold=" << match_threshold;
        logEventThrottled(current_lidar_stamp, "drop_stale_replay", 3.0, "WARN", oss.str());
      }
      replay_index_++;
      continue;
    }
    UwbRangeMeasurement matched_measurement = measurement;
    matched_measurement.time_diff_s = dt;
    measurements.push_back(matched_measurement);
    replay_index_++;
  }
  return measurements;
}

void UwbManager::handleLine(const std::string &line, double stamp)
{
  const auto measurements = filterRepeatedRanges(parseLine(line, stamp), "serial");
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

std::vector<UwbRangeMeasurement> UwbManager::filterRepeatedRanges(const std::vector<UwbRangeMeasurement> &measurements,
                                                                  const std::string &source)
{
  if (!stale_repeat_filter_en_ || measurements.empty()) return measurements;

  std::vector<UwbRangeMeasurement> filtered;
  filtered.reserve(measurements.size());

  for (const auto &measurement : measurements)
  {
    auto &state = repeated_range_states_[measurement.anchor_id];
    const bool same_range = state.valid &&
                            std::fabs(measurement.range_m - state.last_range_m) <= stale_repeat_epsilon_m_;

    if (!same_range)
    {
      state.valid = true;
      state.last_range_m = measurement.range_m;
      state.first_stamp = measurement.stamp;
      state.last_stamp = measurement.stamp;
      state.repeat_count = 1;
      filtered.push_back(measurement);
      continue;
    }

    state.repeat_count++;
    state.last_stamp = measurement.stamp;
    const double repeated_duration = std::max(0.0, state.last_stamp - state.first_stamp);
    const bool repeated_too_many = state.repeat_count > stale_repeat_max_count_;
    const bool repeated_too_long = repeated_duration >= stale_repeat_max_duration_s_;
    if (repeated_too_many && repeated_too_long)
    {
      std::ostringstream oss;
      oss << "DROP_STALE_REPEAT source=" << source
          << " anchor=" << measurement.anchor_id
          << " range=" << measurement.range_m
          << " repeat_count=" << state.repeat_count
          << " duration=" << repeated_duration
          << " epsilon=" << stale_repeat_epsilon_m_;
      logEventThrottled(measurement.stamp,
                        "drop_stale_repeat_" + std::to_string(measurement.anchor_id),
                        3.0, "WARN", oss.str());
      continue;
    }

    filtered.push_back(measurement);
  }

  return filtered;
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
    measurement.time_diff_s = measurement.stamp - now;
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

bool UwbManager::evaluateAnchorFrameResiduals(const std::map<int, UwbAnchor> &candidate_anchors,
                                              const std::vector<UwbAnchorFrameAlignSample> &samples,
                                              double &residual_rms, double &max_abs_residual,
                                              int &valid_range_count, std::vector<int> *used_anchor_ids) const
{
  residual_rms = 0.0;
  max_abs_residual = 0.0;
  valid_range_count = 0;
  double residual_sq_sum = 0.0;
  std::set<int> used_ids;

  for (const auto &sample : samples)
  {
    const auto anchor_it = candidate_anchors.find(sample.anchor_id);
    if (anchor_it == candidate_anchors.end() || !anchor_it->second.enabled) continue;
    if (sample.range_m <= min_range_m_ || sample.range_m >= max_range_m_) continue;

    const double predicted = (sample.tag_position_w - anchor_it->second.position_w).norm();
    if (predicted < 1e-6 || !std::isfinite(predicted)) continue;

    const double residual = predicted - sample.range_m;
    if (!std::isfinite(residual)) continue;
    residual_sq_sum += residual * residual;
    max_abs_residual = std::max(max_abs_residual, std::fabs(residual));
    valid_range_count++;
    used_ids.insert(sample.anchor_id);
  }

  if (valid_range_count <= 0) return false;
  residual_rms = std::sqrt(residual_sq_sum / static_cast<double>(valid_range_count));
  if (used_anchor_ids != nullptr) used_anchor_ids->assign(used_ids.begin(), used_ids.end());
  return true;
}

bool UwbManager::estimateTWorldUwbByRanges(M3D &R_ext_to_w, V3D &t_ext_to_w, double &residual_rms,
                                           double &max_abs_residual, double &residual_rms_before,
                                           int &valid_range_count, double &trajectory_motion,
                                           std::vector<int> &used_anchor_ids, std::string *failure_reason) const
{
  auto fail = [&](const std::string &reason) {
    if (failure_reason != nullptr) *failure_reason = reason;
    return false;
  };

  std::vector<UwbAnchorFrameAlignSample> samples;
  samples.reserve(anchor_frame_align_samples_.size());
  std::set<int> used_ids;
  V3D min_p = V3D::Zero();
  V3D max_p = V3D::Zero();
  bool motion_init = false;

  for (const auto &sample : anchor_frame_align_samples_)
  {
    const auto anchor_it = configured_anchors_.find(sample.anchor_id);
    if (anchor_it == configured_anchors_.end() || !anchor_it->second.enabled) continue;
    if (sample.range_m <= min_range_m_ || sample.range_m >= max_range_m_) continue;
    samples.push_back(sample);
    used_ids.insert(sample.anchor_id);
    if (!motion_init)
    {
      min_p = sample.tag_position_w;
      max_p = sample.tag_position_w;
      motion_init = true;
    }
    else
    {
      min_p = min_p.cwiseMin(sample.tag_position_w);
      max_p = max_p.cwiseMax(sample.tag_position_w);
    }
  }

  valid_range_count = static_cast<int>(samples.size());
  used_anchor_ids.assign(used_ids.begin(), used_ids.end());
  trajectory_motion = motion_init ? (max_p - min_p).norm() : 0.0;
  const double sample_duration = samples.empty() ? 0.0 : std::max(0.0, samples.back().stamp - samples.front().stamp);

  if (static_cast<int>(used_ids.size()) < anchor_frame_align_min_anchors_)
  {
    return fail("wait_not_enough_anchors");
  }
  if (valid_range_count < anchor_frame_align_min_ranges_)
  {
    return fail("wait_not_enough_ranges");
  }
  if (sample_duration < anchor_frame_align_min_duration_s_)
  {
    return fail("wait_not_enough_duration");
  }
  if (trajectory_motion < anchor_frame_align_min_motion_m_)
  {
    return fail("wait_not_enough_motion");
  }

  if (!evaluateAnchorFrameResiduals(configured_anchors_, samples, residual_rms_before,
                                    max_abs_residual, valid_range_count, nullptr))
  {
    return fail("no_valid_residual_before_optimization");
  }

  V3D tag_centroid = V3D::Zero();
  V3D anchor_centroid = V3D::Zero();
  for (const auto &sample : samples)
  {
    tag_centroid += sample.tag_position_w;
  }
  tag_centroid /= static_cast<double>(samples.size());
  for (int id : used_ids)
  {
    anchor_centroid += configured_anchors_.at(id).position_w;
  }
  anchor_centroid /= static_cast<double>(used_ids.size());

  auto makeTransform = [](double yaw, double tx, double ty, M3D &R, V3D &t) {
    R = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
    t << tx, ty, 0.0;
  };

  auto robustCost = [&](double yaw, double tx, double ty) {
    M3D R;
    V3D t;
    makeTransform(yaw, tx, ty, R, t);
    double cost = 0.0;
    constexpr double huber_delta = 1.0;
    for (const auto &sample : samples)
    {
      const V3D anchor_w = R * configured_anchors_.at(sample.anchor_id).position_w + t;
      const double predicted = (sample.tag_position_w - anchor_w).norm();
      if (predicted < 1e-6 || !std::isfinite(predicted)) continue;
      const double r = predicted - sample.range_m;
      const double a = std::fabs(r);
      cost += (a <= huber_delta) ? 0.5 * r * r : huber_delta * (a - 0.5 * huber_delta);
    }
    return cost;
  };

  double best_yaw = 0.0;
  double best_tx = 0.0;
  double best_ty = 0.0;
  double best_cost = std::numeric_limits<double>::infinity();
  constexpr int yaw_grid_count = 72;
  for (int i = 0; i < yaw_grid_count; ++i)
  {
    const double yaw = -M_PI + (2.0 * M_PI * static_cast<double>(i)) / static_cast<double>(yaw_grid_count);
    M3D R;
    V3D t;
    makeTransform(yaw, 0.0, 0.0, R, t);
    const V3D rotated_anchor_centroid = R * anchor_centroid;
    const double tx = tag_centroid.x() - rotated_anchor_centroid.x();
    const double ty = tag_centroid.y() - rotated_anchor_centroid.y();
    const double cost = robustCost(yaw, tx, ty);
    if (cost < best_cost)
    {
      best_cost = cost;
      best_yaw = yaw;
      best_tx = tx;
      best_ty = ty;
    }
  }

  double yaw = best_yaw;
  double tx = best_tx;
  double ty = best_ty;
  constexpr double huber_delta = 1.0;
  for (int iter = 0; iter < 30; ++iter)
  {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    M3D R;
    V3D t;
    makeTransform(yaw, tx, ty, R, t);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    for (const auto &sample : samples)
    {
      const V3D anchor_ext = configured_anchors_.at(sample.anchor_id).position_w;
      const V3D anchor_w = R * anchor_ext + t;
      const V3D diff = sample.tag_position_w - anchor_w;
      const double predicted = diff.norm();
      if (predicted < 1e-6 || !std::isfinite(predicted)) continue;

      const double r = predicted - sample.range_m;
      const double abs_r = std::fabs(r);
      const double weight = abs_r <= huber_delta ? 1.0 : huber_delta / std::max(abs_r, 1e-9);
      const V3D dir = diff / predicted;
      const V3D d_anchor_d_yaw(-s * anchor_ext.x() - c * anchor_ext.y(),
                                c * anchor_ext.x() - s * anchor_ext.y(),
                                0.0);

      Eigen::Vector3d J;
      J << -dir.dot(d_anchor_d_yaw), -dir.x(), -dir.y();
      A += weight * (J * J.transpose());
      b += -weight * J * r;
    }

    Eigen::LDLT<Eigen::Matrix3d> ldlt(A);
    if (ldlt.info() != Eigen::Success) return fail("optimization_decomposition_failed");
    const Eigen::Vector3d delta = ldlt.solve(b);
    if (!delta.allFinite()) return fail("optimization_non_finite_delta");
    yaw += delta(0);
    tx += delta(1);
    ty += delta(2);
    if (delta.norm() < 1e-6) break;
  }

  makeTransform(yaw, tx, ty, R_ext_to_w, t_ext_to_w);
  if (!R_ext_to_w.allFinite() || !t_ext_to_w.allFinite())
  {
    return fail("non_finite_transform");
  }

  std::map<int, UwbAnchor> aligned_anchors;
  for (const auto &item : configured_anchors_)
  {
    if (!item.second.enabled) continue;
    UwbAnchor aligned = item.second;
    aligned.estimated = true;
    aligned.position_w = R_ext_to_w * item.second.position_w + t_ext_to_w;
    aligned_anchors[aligned.id] = aligned;
  }

  if (!evaluateAnchorFrameResiduals(aligned_anchors, samples, residual_rms,
                                    max_abs_residual, valid_range_count, nullptr))
  {
    return fail("no_valid_residual_after_optimization");
  }

  const bool improvement_ok = residual_rms < residual_rms_before * 0.90 ||
                              residual_rms_before - residual_rms > 0.10;
  if (!improvement_ok)
  {
    return fail("residual_not_improved_enough");
  }
  if (residual_rms > anchor_frame_align_success_rms_m_)
  {
    return fail("residual_rms_too_large");
  }
  if (max_abs_residual > anchor_frame_align_success_max_residual_m_)
  {
    return fail("max_abs_residual_too_large");
  }

  return true;
}

bool UwbManager::tryAlignAnchorFrame(const StatesGroup &state,
                                     const std::vector<UwbRangeMeasurement> &measurements)
{
  if (!anchor_frame_align_en_) return false;
  if (anchor_frame_aligned_) return true;
  if (anchor_frame_align_failed_) return false;

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;

  for (const auto &measurement : measurements)
  {
    const auto anchor_it = configured_anchors_.find(measurement.anchor_id);
    if (anchor_it == configured_anchors_.end() || !anchor_it->second.enabled) continue;
    if (measurement.range_m <= min_range_m_ || measurement.range_m >= max_range_m_) continue;

    UwbAnchorFrameAlignSample sample;
    sample.tag_position_w = tag_position_w;
    sample.anchor_id = measurement.anchor_id;
    sample.range_m = measurement.range_m;
    sample.stamp = measurement.stamp;

    if (!anchor_frame_align_candidate_ready_)
    {
      anchor_frame_align_samples_.push_back(sample);
    }
    else
    {
      if (anchor_frame_align_validation_samples_.empty())
      {
        anchor_frame_align_validation_start_stamp_ = measurement.stamp;
      }
      anchor_frame_align_validation_samples_.push_back(sample);
    }
  }

  if (!anchor_frame_align_candidate_ready_)
  {
    M3D R_ext_to_w = M3D::Identity();
    V3D t_ext_to_w = V3D::Zero();
    double residual_rms = 0.0;
    double max_abs_residual = 0.0;
    double residual_rms_before = 0.0;
    int valid_range_count = 0;
    double trajectory_motion = 0.0;
    std::vector<int> used_anchor_ids;
    std::string failure_reason;
    if (!estimateTWorldUwbByRanges(R_ext_to_w, t_ext_to_w, residual_rms, max_abs_residual,
                                   residual_rms_before, valid_range_count, trajectory_motion,
                                   used_anchor_ids, &failure_reason))
    {
      std::ostringstream oss;
      oss << "WAIT_ANCHOR_FRAME_ALIGN align_method=range_yaw_tx_ty"
          << " reason=" << failure_reason
          << " used_anchor_ids=";
      for (size_t i = 0; i < used_anchor_ids.size(); ++i)
      {
        if (i > 0) oss << ",";
        oss << used_anchor_ids[i];
      }
      oss << " valid_range_count=" << valid_range_count
          << " trajectory_motion=" << trajectory_motion
          << " samples=" << anchor_frame_align_samples_.size();
      logEventThrottled(ros::Time::now().toSec(), "anchor_frame_wait_range_optimization", 3.0, "WARN", oss.str());
      return false;
    }

    anchor_frame_align_R_ext_to_w_ = R_ext_to_w;
    anchor_frame_align_t_ext_to_w_ = t_ext_to_w;
    pending_aligned_anchors_.clear();
    for (const auto &item : configured_anchors_)
    {
      if (!item.second.enabled) continue;
      UwbAnchor aligned_anchor = item.second;
      aligned_anchor.estimated = true;
      aligned_anchor.position_w = R_ext_to_w * item.second.position_w + t_ext_to_w;
      pending_aligned_anchors_[aligned_anchor.id] = aligned_anchor;
    }

    const double yaw = std::atan2(R_ext_to_w(1, 0), R_ext_to_w(0, 0));
    std::ostringstream oss;
    oss << "ANCHOR_FRAME_ALIGN_CANDIDATE align_method=range_yaw_tx_ty"
        << " used_anchor_ids=";
    for (size_t i = 0; i < used_anchor_ids.size(); ++i)
    {
      if (i > 0) oss << ",";
      oss << used_anchor_ids[i];
    }
    oss << " valid_range_count=" << valid_range_count
        << " trajectory_motion=" << trajectory_motion
        << " yaw=" << yaw
        << " translation=" << t_ext_to_w.transpose()
        << " residual_before=" << residual_rms_before
        << " residual_rms=" << residual_rms
        << " max_abs_residual=" << max_abs_residual;
    logEvent(ros::Time::now().toSec(), "INFO", oss.str());

    for (const auto &item : pending_aligned_anchors_)
    {
      std::ostringstream anchor_oss;
      anchor_oss << "ANCHOR_FRAME_ALIGN_CANDIDATE_ANCHOR id=" << item.first
                 << " manual=" << configured_anchors_[item.first].position_w.transpose()
                 << " aligned=" << item.second.position_w.transpose();
      logEvent(ros::Time::now().toSec(), "INFO", anchor_oss.str());
    }

    anchor_frame_align_candidate_ready_ = true;
    anchor_frame_align_validation_samples_.clear();
    anchor_frame_align_validation_start_stamp_ = 0.0;
    return false;
  }

  const double validation_duration = anchor_frame_align_validation_samples_.empty() ? 0.0 :
      std::max(0.0, anchor_frame_align_validation_samples_.back().stamp - anchor_frame_align_validation_start_stamp_);
  double validation_rms = 0.0;
  double validation_max_abs = 0.0;
  int validation_count = 0;
  std::vector<int> validation_anchor_ids;
  evaluateAnchorFrameResiduals(pending_aligned_anchors_, anchor_frame_align_validation_samples_,
                               validation_rms, validation_max_abs, validation_count, &validation_anchor_ids);

  if (validation_duration < anchor_frame_align_validation_duration_s_)
  {
    std::ostringstream oss;
    oss << "ANCHOR_FRAME_ALIGN_VALIDATING duration=" << validation_duration
        << " required_duration=" << anchor_frame_align_validation_duration_s_
        << " valid_range_count=" << validation_count
        << " residual_rms=" << validation_rms
        << " max_abs_residual=" << validation_max_abs;
    logEventThrottled(ros::Time::now().toSec(), "anchor_frame_align_validating", 2.0, "INFO", oss.str());
    return false;
  }

  if (validation_rms > anchor_frame_align_success_rms_m_ ||
      validation_max_abs > anchor_frame_align_success_max_residual_m_)
  {
    std::ostringstream oss;
    oss << "ANCHOR_FRAME_ALIGN_FAILED reason=validation_residual_too_large"
        << " residual_rms=" << validation_rms
        << " max_abs_residual=" << validation_max_abs
        << " valid_range_count=" << validation_count
        << " used_anchor_ids=";
    for (size_t i = 0; i < validation_anchor_ids.size(); ++i)
    {
      if (i > 0) oss << ",";
      oss << validation_anchor_ids[i];
    }
    logEvent(ros::Time::now().toSec(), "ERROR", oss.str());
    for (const auto &item : pending_aligned_anchors_)
    {
      std::ostringstream anchor_oss;
      anchor_oss << "ANCHOR_FRAME_ALIGN_FAILED_ANCHOR id=" << item.first
                 << " manual=" << configured_anchors_[item.first].position_w.transpose()
                 << " aligned=" << item.second.position_w.transpose();
      logEvent(ros::Time::now().toSec(), "ERROR", anchor_oss.str());
    }
    pending_aligned_anchors_.clear();
    anchors_.clear();
    anchor_frame_align_failed_ = true;
    anchor_frame_align_en_ = false;
    return false;
  }

  anchors_ = pending_aligned_anchors_;
  pending_aligned_anchors_.clear();
  anchor_frame_aligned_ = !anchors_.empty();
  std::ostringstream oss;
  oss << "ANCHOR_FRAME_ALIGN_SUCCESS validation_residual_rms=" << validation_rms
      << " validation_max_abs_residual=" << validation_max_abs
      << " valid_range_count=" << validation_count
      << " translation=" << anchor_frame_align_t_ext_to_w_.transpose()
      << " yaw=" << std::atan2(anchor_frame_align_R_ext_to_w_(1, 0), anchor_frame_align_R_ext_to_w_(0, 0));
  logEvent(ros::Time::now().toSec(), "INFO", oss.str());
  for (const auto &item : anchors_)
  {
    std::ostringstream anchor_oss;
    anchor_oss << "ANCHOR_FRAME_ALIGNED_ANCHOR id=" << item.first
               << " position=" << item.second.position_w.transpose();
    logEvent(ros::Time::now().toSec(), "INFO", anchor_oss.str());
  }
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
  if (!en_) return 0;
  const std::string source = toLower(input_source_);
  if (source == "file" || source == "txt" || source == "replay")
  {
    ROS_WARN_THROTTLE(3.0,
                      "[UWB] File replay requires SLAM/LiDAR timestamps; skip wall-time applyRangeUpdate().");
    logEventThrottled(ros::Time::now().toSec(), "skip_wall_time_replay", 3.0, "WARN",
                      "SKIP_WALL_TIME_REPLAY reason=file_replay_requires_applyRangeUpdateAt");
    return 0;
  }

  const double now = ros::Time::now().toSec();
  return applyRangeUpdateAt(state, now, now);
}

int UwbManager::applyRangeUpdateAt(StatesGroup &state, double current_lidar_stamp, double lidar_start_stamp)
{
  if (!en_) return 0;
  const double now = current_lidar_stamp;
  const std::string source = toLower(input_source_);
  const auto measurements = (source == "file" || source == "txt" || source == "replay") ?
                            takeReplayMeasurements(current_lidar_stamp, lidar_start_stamp) :
                            takeRecentMeasurements(now);
  if (measurements.empty()) return 0;

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
  const double now = ros::Time::now().toSec();
  const int required_anchors = std::max(2, min_update_anchors_);
  std::map<int, UwbRangeMeasurement> latest_by_anchor;
  for (const auto &measurement : measurements)
  {
    if (anchors_.find(measurement.anchor_id) == anchors_.end()) continue;
    latest_by_anchor[measurement.anchor_id] = measurement;
  }
  if (static_cast<int>(latest_by_anchor.size()) < required_anchors)
  {
    std::ostringstream oss;
    oss << "UWB_UPDATE action=skip_not_enough_anchors used=" << latest_by_anchor.size()
        << " required=" << required_anchors;
    logEventThrottled(now, "skip_not_enough_anchors", 1.0, "WARN", oss.str());
    return 0;
  }

  std::vector<UwbRangeMeasurement> usable_measurements;
  usable_measurements.reserve(latest_by_anchor.size());
  for (const auto &item : latest_by_anchor)
  {
    usable_measurements.push_back(item.second);
  }
  if (prefer_anchors_ > 0 && static_cast<int>(usable_measurements.size()) > prefer_anchors_)
  {
    usable_measurements.resize(prefer_anchors_);
  }

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(usable_measurements.size(), DIM_STATE);
  Eigen::MatrixXd H_tag = Eigen::MatrixXd::Zero(usable_measurements.size(), 3);
  Eigen::VectorXd z = Eigen::VectorXd::Zero(usable_measurements.size());

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;
  int row = 0;
  double residual_sq_sum = 0.0;
  double max_abs_residual = 0.0;
  double max_abs_time_diff = 0.0;
  double time_diff_for_log = 0.0;
  std::vector<int> used_anchor_ids;
  std::vector<std::string> detail_logs;
  for (const auto &measurement : usable_measurements)
  {
    const auto anchor_it = anchors_.find(measurement.anchor_id);
    if (anchor_it == anchors_.end()) continue;

    const V3D diff = tag_position_w - anchor_it->second.position_w;
    const double predicted_3d = diff.norm();
    const double predicted_xy = std::hypot(diff.x(), diff.y());
    const double height_diff = diff.z();
    const double predicted_range = use_3d_range_model_ ? predicted_3d : predicted_xy;
    if (predicted_range < 1e-6) continue;

    const double residual = predicted_range - measurement.range_m;
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
            << " predicted=" << predicted_range;
        logEventThrottled(ros::Time::now().toSec(),
                          "reject_range_" + std::to_string(measurement.anchor_id),
                          2.0, "WARN", oss.str());
      }
      continue;
    }

    V3D direction = V3D::Zero();
    if (use_3d_range_model_)
    {
      direction = diff / predicted_3d;
    }
    else if (predicted_xy > 1e-6)
    {
      direction << diff.x() / predicted_xy, diff.y() / predicted_xy, 0.0;
    }
    else
    {
      continue;
    }

    if (update_orientation_ && !update_xy_only_)
    {
      H.block<1, 3>(row, 0) = direction.transpose() * (-state.rot_end * skewSymmetric(tag_offset_used));
    }
    H(row, 3) = direction.x();
    H(row, 4) = direction.y();
    H(row, 5) = (update_z_ && !update_xy_only_) ? direction.z() : 0.0;
    if (tag_offset_estimate_en_ && !update_xy_only_)
    {
      H_tag.block<1, 3>(row, 0) = direction.transpose() * state.rot_end;
    }
    z(row) = -residual;
    residual_sq_sum += residual * residual;
    max_abs_residual = std::max(max_abs_residual, std::fabs(residual));
    if (std::fabs(measurement.time_diff_s) > max_abs_time_diff)
    {
      max_abs_time_diff = std::fabs(measurement.time_diff_s);
      time_diff_for_log = measurement.time_diff_s;
    }
    used_anchor_ids.push_back(measurement.anchor_id);
    {
      std::ostringstream detail;
      detail << "UWB_RANGE anchor=" << measurement.anchor_id
             << " slam_time=" << now
             << " measurement_time=" << measurement.stamp
             << " anchor_xyz=" << anchor_it->second.position_w.transpose()
             << " tag_xyz=" << tag_position_w.transpose()
             << " measured=" << measurement.range_m
             << " predicted_3d=" << predicted_3d
             << " predicted_xy=" << predicted_xy
             << " height_diff=" << height_diff
             << " residual=" << residual
             << " time_diff=" << measurement.time_diff_s
             << " range_model=" << (use_3d_range_model_ ? "3d" : "legacy_xy");
      detail_logs.push_back(detail.str());
    }
    row++;
  }

  if (row < required_anchors)
  {
    std::ostringstream oss;
    oss << "UWB_UPDATE action=skip_not_enough_anchors used=" << row
        << " required=" << required_anchors;
    logEventThrottled(now, "skip_not_enough_anchors_after_gate", 1.0, "WARN", oss.str());
    return 0;
  }
  H.conservativeResize(row, DIM_STATE);
  H_tag.conservativeResize(row, 3);
  z.conservativeResize(row);
  const bool two_anchor_case = row == 2;
  const bool two_anchor_update_disabled = two_anchor_case && min_anchors_for_update_ > 2;
  const double effective_range_noise_m = range_noise_m_ * (two_anchor_case ? two_anchor_sigma_scale_ : 1.0);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(row, row) * (effective_range_noise_m * effective_range_noise_m);
  const double residual_rms = std::sqrt(residual_sq_sum / static_cast<double>(row));
  const bool h_orientation_zero = H.block(0, 0, row, 3).cwiseAbs().maxCoeff() < 1e-12;
  const bool h_z_zero = H.col(5).cwiseAbs().maxCoeff() < 1e-12;

  const bool estimate_tag_offset_this_update =
      !update_xy_only_ && tag_offset_estimate_en_ && row >= tag_offset_estimate_min_anchors_;

  if (!estimate_tag_offset_this_update)
  {
    if (tag_offset_estimate_en_ && update_xy_only_)
    {
      logEventThrottled(now, "skip_tag_offset_xy_only", 3.0, "WARN",
                        "SKIP_TAG_OFFSET_ESTIMATION reason=xy_only_update");
    }
    else if (tag_offset_estimate_en_)
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
      const int pos_dims = update_z_ ? 3 : 2;
      for (int i = 0; i < pos_dims; ++i)
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
    Eigen::VectorXd dx_unlimited = K * z;
    if (dx_unlimited.size() != DIM_STATE || !dx_unlimited.allFinite()) return 0;
    const double z_correction_before_clamp = dx_unlimited(5, 0);

    Eigen::VectorXd dx_raw_dynamic = K * z;
    if (dx_raw_dynamic.size() != DIM_STATE || !dx_raw_dynamic.allFinite()) return 0;

    VD(DIM_STATE) dx_raw = VD(DIM_STATE)::Zero();
    dx_raw = dx_raw_dynamic;

    V3D trans_raw = dx_raw.block<3, 1>(3, 0);
    const double xy_correction_raw = std::hypot(trans_raw.x(), trans_raw.y());
    std::ostringstream used_ids;
    for (size_t i = 0; i < used_anchor_ids.size(); ++i)
    {
      if (i > 0) used_ids << ",";
      used_ids << used_anchor_ids[i];
    }
    auto log_action = [&](const std::string &level, const std::string &action,
                          double xy_correction_applied, double clamp_ratio, const V3D &delta) {
      std::ostringstream oss;
      oss << "UWB_UPDATE action=" << action
          << " uwb_state=" << uwbStateName(uwb_state_)
          << " used_anchor_ids=" << used_ids.str()
          << " residual_rms=" << residual_rms
          << " max_abs_residual=" << max_abs_residual
          << " xy_correction_raw=" << xy_correction_raw
          << " xy_correction_applied=" << xy_correction_applied
          << " clamp_ratio=" << clamp_ratio
          << " time_diff=" << time_diff_for_log
          << " max_abs_time_diff=" << max_abs_time_diff
          << " consecutive_good_count=" << uwb_consecutive_good_count_
          << " update_enable=" << static_cast<int>(update_en_)
          << " residual_debug_only=" << static_cast<int>(residual_debug_only_)
          << " two_anchor_update_disabled=" << static_cast<int>(two_anchor_update_disabled)
          << " two_anchor_baseline_mode=" << static_cast<int>(two_anchor_baseline_mode_)
          << " suspect_hold_disabled=" << static_cast<int>(!suspect_hold_en_)
          << " lost_hold_disabled=" << static_cast<int>(!lost_hold_en_)
          << " H_orientation_zero=" << static_cast<int>(h_orientation_zero)
          << " H_z_zero=" << static_cast<int>(h_z_zero)
          << " effective_sigma=" << effective_range_noise_m
          << " z_correction_before_clamp=" << z_correction_before_clamp
          << " z_correction_after_clamp=" << delta.z()
          << " trans_add=" << delta.transpose()
          << " update_xy_only=" << static_cast<int>(update_xy_only_)
          << " update_z=" << static_cast<int>(update_z_)
          << " update_orientation=" << static_cast<int>(update_orientation_);
      logEvent(now, level, oss.str());
      for (const auto &detail : detail_logs) logEvent(now, "INFO", detail);
    };

    const bool dry_run = residual_debug_only_ || !update_en_;
    if (dry_run)
    {
      log_action("INFO", "dry_run", 0.0, 1.0, V3D::Zero());
      return 0;
    }

    const double normal_residual_rms = max_residual_rms_ > 0.0 ? max_residual_rms_ : 0.5;
    constexpr double lost_residual_rms = 2.0;
    constexpr double relocalize_residual_rms = 3.0;
    constexpr int lost_recovery_good_updates = 5;

    if (two_anchor_update_disabled)
    {
      log_action("INFO", "two_anchor_dry_run", 0.0, 1.0, V3D::Zero());
      return 0;
    }

    if (residual_rms >= relocalize_residual_rms)
    {
      uwb_state_ = 2;
      uwb_consecutive_good_count_ = 0;
      uwb_lost_good_count_ = 0;
      uwb_consecutive_gate_ready_ = false;
      log_action("WARN", "need_relocalize_or_hold", 0.0, 1.0, V3D::Zero());
      return 0;
    }

    if (lost_hold_en_ && uwb_state_ == 2)
    {
      if (residual_rms < normal_residual_rms)
      {
        uwb_lost_good_count_++;
      }
      else
      {
        uwb_lost_good_count_ = 0;
      }
      if (uwb_lost_good_count_ >= lost_recovery_good_updates)
      {
        uwb_state_ = 0;
        uwb_lost_good_count_ = 0;
        uwb_consecutive_good_count_ = 0;
        log_action("INFO", "lost_recovered_wait", 0.0, 1.0, V3D::Zero());
      }
      else
      {
        log_action("WARN", "lost_hold", 0.0, 1.0, V3D::Zero());
      }
      return 0;
    }

    if (residual_rms >= lost_residual_rms)
    {
      uwb_state_ = 2;
      uwb_consecutive_good_count_ = 0;
      uwb_lost_good_count_ = 0;
      if (lost_hold_en_)
      {
        log_action("WARN", "lost_hold", 0.0, 1.0, V3D::Zero());
        return 0;
      }
    }
    else if (residual_rms >= normal_residual_rms)
    {
      uwb_state_ = 1;
      uwb_consecutive_good_count_ = 0;
      uwb_lost_good_count_ = 0;
      if (suspect_hold_en_)
      {
        log_action("WARN", "suspect_hold", 0.0, 1.0, V3D::Zero());
        return 0;
      }
    }
    else
    {
      uwb_state_ = 0;
      uwb_lost_good_count_ = 0;
    }

    constexpr double limited_update_max_residual_rms = 2.0;
    constexpr double limited_update_max_abs_residual = 3.0;
    constexpr double limited_update_max_xy_raw = 3.0;
    const bool large_correction = xy_correction_raw > max_xy_correction_normal_;
    const bool allow_limited_large_correction =
        large_correction &&
        row >= 2 &&
        residual_rms < limited_update_max_residual_rms &&
        max_abs_residual < limited_update_max_abs_residual &&
        xy_correction_raw < limited_update_max_xy_raw;
    if (large_correction && !allow_limited_large_correction)
    {
      uwb_consecutive_good_count_ = 0;
      log_action("WARN", "large_correction_hold", 0.0, 1.0, V3D::Zero());
      return 0;
    }

    if (!allow_limited_large_correction &&
        !uwb_consecutive_gate_ready_ && require_consecutive_good_updates_ > 0)
    {
      if (residual_rms < good_residual_rms_)
      {
        uwb_consecutive_good_count_++;
      }
      else
      {
        uwb_consecutive_good_count_ = 0;
      }
      if (uwb_consecutive_good_count_ < require_consecutive_good_updates_)
      {
        log_action("INFO", "wait_consecutive_good", 0.0, 1.0, V3D::Zero());
        return 0;
      }
      uwb_consecutive_gate_ready_ = true;
    }

    double clamp_ratio = 1.0;
    if (max_update_step_xy_ > 0.0 && xy_correction_raw > max_update_step_xy_)
    {
      clamp_ratio = max_update_step_xy_ / std::max(xy_correction_raw, 1e-9);
    }

    const Eigen::MatrixXd K_apply = K * clamp_ratio;
    Eigen::VectorXd dx_dynamic = K_apply * z;
    if (dx_dynamic.size() != DIM_STATE || !dx_dynamic.allFinite()) return 0;

    VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
    dx = dx_dynamic;

    V3D rot_add = dx.block<3, 1>(0, 0);
    V3D trans_add = dx.block<3, 1>(3, 0);
    const double xy_correction_applied = std::hypot(trans_add.x(), trans_add.y());

    state += dx;
    const MD(DIM_STATE, DIM_STATE) I_STATE = MD(DIM_STATE, DIM_STATE)::Identity();
    const MD(DIM_STATE, DIM_STATE) I_KH = I_STATE - K_apply * H;
    state.cov = I_KH * cov_for_uwb * I_KH.transpose() + K_apply * R * K_apply.transpose();
    state.cov = 0.5 * (state.cov + state.cov.transpose());
    snapStateForDeterminism(state);

    const char *update_action = allow_limited_large_correction ? "xy_update_limited" :
                                (two_anchor_case ? "xy_update_weak_2anchors" : "xy_update");
    log_action("INFO", update_action,
               xy_correction_applied, clamp_ratio, trans_add);
    logUpdate(now, row, z.norm(), rot_add, trans_add, V3D::Zero());
    ROS_INFO_THROTTLE(1.0, "[UWB] xy_update used=%d residual_rms=%.3f raw_xy=%.4f applied_xy=%.4f clamp=%.3f trans_add=[%.4f %.4f %.4f]",
                      row, residual_rms, xy_correction_raw, xy_correction_applied, clamp_ratio,
                      trans_add.x(), trans_add.y(), trans_add.z());
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

  logUpdate(ros::Time::now().toSec(), row, z.norm(), rot_add, trans_add, tag_offset_add);
  ROS_INFO_THROTTLE(1.0, "[UWB] EKF update used=%d residual_norm=%.3f trans_add=%.4f m tag_offset=[%.3f %.3f %.3f]",
                    row, z.norm(), trans_add.norm(),
                    tag_offset_est_body_.x(), tag_offset_est_body_.y(), tag_offset_est_body_.z());
  return row;
}
