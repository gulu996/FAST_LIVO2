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
#include <tuple>
#include <unistd.h>

const char *uwbUpdateStatusName(UwbUpdateStatus status)
{
  switch (status)
  {
    case UwbUpdateStatus::UPDATED: return "UPDATED";
    case UwbUpdateStatus::NOT_UPDATED: return "NOT_UPDATED";
    case UwbUpdateStatus::WAITING_INITIALIZATION: return "WAITING_INITIALIZATION";
    case UwbUpdateStatus::DEBUG_ONLY: return "DEBUG_ONLY";
  }
  return "NOT_UPDATED";
}

const char *uwbUpdateOutcomeName(UwbUpdateOutcome outcome)
{
  switch (outcome)
  {
    case UwbUpdateOutcome::ACCEPTED: return "ACCEPTED";
    case UwbUpdateOutcome::REJECTED: return "REJECTED";
    case UwbUpdateOutcome::SKIPPED: return "SKIPPED";
    case UwbUpdateOutcome::WAITING: return "WAITING";
  }
  return "SKIPPED";
}

const char *uwbRejectReasonName(UwbRejectReason reason)
{
  switch (reason)
  {
    case UwbRejectReason::NONE: return "NONE";
    case UwbRejectReason::UPDATE_DISABLED: return "UPDATE_DISABLED";
    case UwbRejectReason::DEBUG_ONLY: return "DEBUG_ONLY";
    case UwbRejectReason::BASELINE_NOT_INITIALIZED: return "BASELINE_NOT_INITIALIZED";
    case UwbRejectReason::ANCHORS_NOT_READY: return "ANCHORS_NOT_READY";
    case UwbRejectReason::NOT_ENOUGH_VALID_ANCHORS: return "NOT_ENOUGH_VALID_ANCHORS";
    case UwbRejectReason::INVALID_RANGE_STATUS: return "INVALID_RANGE_STATUS";
    case UwbRejectReason::INVALID_RAW_RANGE: return "INVALID_RAW_RANGE";
    case UwbRejectReason::INVALID_CORRECTED_RANGE: return "INVALID_CORRECTED_RANGE";
    case UwbRejectReason::UNKNOWN_ANCHOR: return "UNKNOWN_ANCHOR";
    case UwbRejectReason::TIMESTAMP_INVALID: return "TIMESTAMP_INVALID";
    case UwbRejectReason::PAIR_TIME_MISMATCH: return "PAIR_TIME_MISMATCH";
    case UwbRejectReason::RANGE_LIMIT: return "RANGE_LIMIT";
    case UwbRejectReason::RANGE_RESIDUAL_GATE: return "RANGE_RESIDUAL_GATE";
    case UwbRejectReason::RANGE_JUMP: return "RANGE_JUMP";
    case UwbRejectReason::RESIDUAL_JUMP: return "RESIDUAL_JUMP";
    case UwbRejectReason::NIS_GATE: return "NIS_GATE";
    case UwbRejectReason::LOW_GEOMETRY: return "LOW_GEOMETRY";
    case UwbRejectReason::TWO_ANCHOR_RESIDUAL_GATE: return "TWO_ANCHOR_RESIDUAL_GATE";
    case UwbRejectReason::BASELINE_CONSISTENCY_GATE: return "BASELINE_CONSISTENCY_GATE";
    case UwbRejectReason::SINGLE_ANCHOR_NOT_CONFIRMED: return "SINGLE_ANCHOR_NOT_CONFIRMED";
    case UwbRejectReason::NEAR_ANCHOR_DISABLED: return "NEAR_ANCHOR_DISABLED";
    case UwbRejectReason::CORRIDOR_DIRECTION_GATE: return "CORRIDOR_DIRECTION_GATE";
    case UwbRejectReason::CORRECTION_TOO_SMALL: return "CORRECTION_TOO_SMALL";
    case UwbRejectReason::CORRECTION_CLAMPED: return "CORRECTION_CLAMPED";
    case UwbRejectReason::COVARIANCE_INVALID: return "COVARIANCE_INVALID";
    case UwbRejectReason::NON_FINITE_CORRECTION: return "NON_FINITE_CORRECTION";
  }
  return "NONE";
}

namespace
{
std::string formatAnchorIds(const std::vector<int> &ids)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < ids.size(); ++i)
  {
    if (i > 0) oss << ",";
    oss << ids[i];
  }
  oss << "]";
  return oss.str();
}
} // namespace

std::string formatUwbResultLine(const UwbUpdateReport &report)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3)
      << "[UWB_RESULT] attempt=" << report.attempt_id
      << " status=" << uwbUpdateStatusName(report.status)
      << " outcome=" << uwbUpdateOutcomeName(report.outcome)
      << " mode=" << report.mode;
  if (report.state_updated)
  {
    oss << " anchors=" << formatAnchorIds(report.used_anchor_ids)
        << " position_before=(" << report.system_position_before.x() << ","
        << report.system_position_before.y() << "," << report.system_position_before.z() << ")m"
        << " delta=(" << std::showpos << report.applied_position_correction.x() << ","
        << report.applied_position_correction.y() << "," << report.applied_position_correction.z()
        << std::noshowpos << ")m"
        << " position_after=(" << report.system_position_after.x() << ","
        << report.system_position_after.y() << "," << report.system_position_after.z() << ")m"
        << " delta_norm=" << report.correction_norm << "m"
        << " clamped=" << static_cast<int>(report.correction_clamped);
  }
  else
  {
    if (!report.received_anchor_ids.empty())
      oss << " received_anchors=" << formatAnchorIds(report.received_anchor_ids);
    oss << " valid_anchors=" << formatAnchorIds(report.used_anchor_ids)
        << " rejected_anchors=" << formatAnchorIds(report.rejected_anchor_ids)
        << " reason=" << uwbRejectReasonName(report.primary_reason);
    if (report.required_anchor_count > 0)
      oss << " required=" << report.required_anchor_count << " valid=" << report.valid_anchor_count;
    if (report.required_motion_m > 0.0)
      oss << " current_motion=" << report.current_motion_m
          << "m required_motion=" << report.required_motion_m << "m";
    oss << " position_unchanged=(" << report.system_position_after.x() << ","
        << report.system_position_after.y() << "," << report.system_position_after.z() << ")m"
        << " state_updated=0 covariance_updated=" << static_cast<int>(report.covariance_updated);
  }
  return oss.str();
}

Eigen::MatrixXd applyUwbUpdateMaskAndProjection(const Eigen::MatrixXd &gain,
                                                bool allow_z,
                                                bool allow_orientation,
                                                const V3D *position_projection_direction)
{
  Eigen::MatrixXd used_gain = gain;
  for (int row = 0; row < used_gain.rows(); ++row)
  {
    const bool orientation = row >= 0 && row < 3;
    const bool position_xy = row == 3 || row == 4;
    const bool position_z = row == 5;
    if ((orientation && allow_orientation) || position_xy || (position_z && allow_z)) continue;
    used_gain.row(row).setZero();
  }

  if (position_projection_direction != nullptr && used_gain.rows() >= 6)
  {
    V3D direction = *position_projection_direction;
    if (!allow_z) direction.z() = 0.0;
    const double norm = direction.norm();
    if (norm > 1e-12)
    {
      direction /= norm;
      const Eigen::RowVectorXd along = direction.x() * used_gain.row(3) +
                                       direction.y() * used_gain.row(4) +
                                       direction.z() * used_gain.row(5);
      used_gain.row(3) = direction.x() * along;
      used_gain.row(4) = direction.y() * along;
      used_gain.row(5) = direction.z() * along;
    }
  }
  return used_gain;
}

bool computeUwbJosephCovariance(const Eigen::MatrixXd &prior_covariance,
                                const Eigen::MatrixXd &measurement_jacobian,
                                const Eigen::MatrixXd &measurement_covariance,
                                const Eigen::MatrixXd &used_gain,
                                Eigen::MatrixXd &updated_covariance,
                                double &max_asymmetry,
                                double &min_diagonal)
{
  if (prior_covariance.rows() != prior_covariance.cols() ||
      used_gain.rows() != prior_covariance.rows() ||
      measurement_jacobian.cols() != prior_covariance.cols() ||
      used_gain.cols() != measurement_jacobian.rows() ||
      measurement_covariance.rows() != used_gain.cols() ||
      measurement_covariance.cols() != used_gain.cols())
  {
    return false;
  }
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(prior_covariance.rows(), prior_covariance.cols());
  const Eigen::MatrixXd i_kh = identity - used_gain * measurement_jacobian;
  updated_covariance = i_kh * prior_covariance * i_kh.transpose() +
                       used_gain * measurement_covariance * used_gain.transpose();
  if (!updated_covariance.allFinite()) return false;
  max_asymmetry = (updated_covariance - updated_covariance.transpose()).cwiseAbs().maxCoeff();
  const double covariance_scale = std::max(1.0, updated_covariance.cwiseAbs().maxCoeff());
  updated_covariance = 0.5 * (updated_covariance + updated_covariance.transpose());
  min_diagonal = updated_covariance.diagonal().minCoeff();
  return std::isfinite(max_asymmetry) && std::isfinite(min_diagonal) &&
         max_asymmetry <= 1e-8 * covariance_scale && min_diagonal >= -1e-10;
}

bool correctUwbRangeValue(double raw_range_m, double range_bias_m,
                          double &corrected_range_m, UwbRejectReason &reject_reason)
{
  corrected_range_m = std::numeric_limits<double>::quiet_NaN();
  if (!std::isfinite(raw_range_m) || raw_range_m <= 0.0)
  {
    reject_reason = UwbRejectReason::INVALID_RAW_RANGE;
    return false;
  }
  if (!std::isfinite(range_bias_m))
  {
    reject_reason = UwbRejectReason::INVALID_RANGE_STATUS;
    return false;
  }

  corrected_range_m = raw_range_m - range_bias_m;
  if (!std::isfinite(corrected_range_m) || corrected_range_m <= 0.0)
  {
    reject_reason = UwbRejectReason::INVALID_CORRECTED_RANGE;
    return false;
  }
  reject_reason = UwbRejectReason::NONE;
  return true;
}

double selectUwbPositionCovFloor(double normal_floor_m, double degraded_floor_m,
                                 bool degraded_only, bool is_degraded)
{
  if (is_degraded || !degraded_only) return std::max(normal_floor_m, degraded_floor_m);
  return normal_floor_m;
}

bool isUwbReplayCrossFormatDuplicate(const UwbRangeMeasurement &debug_measurement,
                                     const UwbRangeMeasurement &distance_measurement,
                                     double max_time_difference_s)
{
  if (debug_measurement.source_format != "uwbdbg" ||
      distance_measurement.source_format != "distance" ||
      debug_measurement.anchor_id != distance_measurement.anchor_id ||
      !std::isfinite(debug_measurement.raw_range_m) ||
      !std::isfinite(distance_measurement.raw_range_m))
    return false;
  const int64_t debug_range_mm =
      static_cast<int64_t>(std::llround(debug_measurement.raw_range_m * 1000.0));
  const int64_t distance_range_mm =
      static_cast<int64_t>(std::llround(distance_measurement.raw_range_m * 1000.0));
  return debug_range_mm == distance_range_mm &&
         std::fabs(debug_measurement.stamp - distance_measurement.stamp) <=
             std::max(0.0, max_time_difference_s);
}

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

bool xmlRpcToString(const XmlRpc::XmlRpcValue &value, std::string &out)
{
  if (value.getType() != XmlRpc::XmlRpcValue::TypeString) return false;
  out = static_cast<std::string>(value);
  return true;
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

UwbRejectReason rejectReasonFromLegacy(const std::string &reason, const std::string &action)
{
  if (reason == "none")
  {
    if (action == "dry_run" || action == "two_anchor_dry_run") return UwbRejectReason::DEBUG_ONLY;
    if (action == "skip_not_enough_anchors") return UwbRejectReason::NOT_ENOUGH_VALID_ANCHORS;
    if (action == "wait_single_anchor_confirm") return UwbRejectReason::SINGLE_ANCHOR_NOT_CONFIRMED;
    if (action.find("wait") != std::string::npos || action.find("hold") != std::string::npos ||
        action.find("reject") != std::string::npos || action.find("relocal") != std::string::npos)
      return UwbRejectReason::RANGE_RESIDUAL_GATE;
    return UwbRejectReason::NONE;
  }
  if (reason == "time_mismatch") return UwbRejectReason::PAIR_TIME_MISMATCH;
  if (reason == "baseline_not_initialized" || reason == "baseline_direction_invalid")
    return UwbRejectReason::BASELINE_NOT_INITIALIZED;
  if (reason == "range_out_of_bounds") return UwbRejectReason::RANGE_LIMIT;
  if (reason == "range_jump" || reason == "range_speed") return UwbRejectReason::RANGE_JUMP;
  if (reason == "residual_jump") return UwbRejectReason::RESIDUAL_JUMP;
  if (reason == "near_anchor") return UwbRejectReason::NEAR_ANCHOR_DISABLED;
  if (reason == "turn_or_corner") return UwbRejectReason::CORRIDOR_DIRECTION_GATE;
  if (reason == "wait_confirm" || reason == "branch_ambiguous")
    return UwbRejectReason::SINGLE_ANCHOR_NOT_CONFIRMED;
  if (reason == "baseline_consistency_error_large") return UwbRejectReason::BASELINE_CONSISTENCY_GATE;
  if (reason == "two_anchor_residual_gate" || reason == "two_anchor_residual_large")
    return UwbRejectReason::TWO_ANCHOR_RESIDUAL_GATE;
  if (reason == "bad_anchor_geometry" || reason == "anchor_not_on_baseline" ||
      reason == "uwb_only_solve_failed") return UwbRejectReason::LOW_GEOMETRY;
  if (reason.find("residual") != std::string::npos || reason.find("correction") != std::string::npos ||
      reason == "hard_reject_xy_raw") return UwbRejectReason::RANGE_RESIDUAL_GATE;
  if (reason == "two_anchor_policy_dry_run") return UwbRejectReason::DEBUG_ONLY;
  if (reason == "two_anchor_policy_disable") return UwbRejectReason::UPDATE_DISABLED;
  return UwbRejectReason::INVALID_RANGE_STATUS;
}

bool actionIsWaiting(const std::string &action)
{
  return action.find("wait") != std::string::npos || action.find("hold") != std::string::npos;
}
} // namespace

UwbManager::UwbManager() = default;

UwbManager::~UwbManager()
{
  shutdown();
}

bool UwbManager::correctUwbRange(int anchor_id, double raw_range_m,
                                 double &corrected_range_m, double &range_bias_m,
                                 UwbRejectReason &reject_reason) const
{
  range_bias_m = 0.0;
  const auto configured_it = configured_anchors_.find(anchor_id);
  if (configured_it != configured_anchors_.end())
  {
    range_bias_m = configured_it->second.range_bias_m;
  }
  else
  {
    const auto anchor_it = anchors_.find(anchor_id);
    if (anchor_it != anchors_.end()) range_bias_m = anchor_it->second.range_bias_m;
  }

  if (!correctUwbRangeValue(raw_range_m, range_bias_m, corrected_range_m, reject_reason))
    return false;
  if (corrected_range_m < min_range_m_ || corrected_range_m > max_range_m_)
  {
    reject_reason = UwbRejectReason::RANGE_LIMIT;
    return false;
  }
  return true;
}

bool UwbManager::initialize(ros::NodeHandle &nh, const std::string &save_path)
{
  if (!loadParameters(nh) || !en_) return false;

  const std::string source = toLower(input_source_);
  const bool replay_source = source == "file" || source == "txt" || source == "replay";
  const std::string raw_log_path = save_path + log_filename_;
  const std::string update_log_path = save_path + update_log_filename_;
  std::ifstream existing_update_log(update_log_path, std::ios::binary | std::ios::ate);
  const bool update_log_is_empty = !existing_update_log.is_open() || existing_update_log.tellg() <= 0;
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    update_log_file_.open(update_log_path, std::ios::out | std::ios::app);
    if (update_log_file_.is_open())
    {
      if (update_log_is_empty)
      {
        update_log_file_ << "# UWB_UPDATE_LOG_VERSION=2\n"
                         << "# All lines belonging to one update share the same attempt id.\n"
                         << "# UWB_RESULT is always written before detailed debug lines.\n"
                         << "# Position unit: meter.\n"
                         << "# Time unit: second.\n";
      }
      update_log_file_ << "================ UWB SESSION START ================\n"
                       << "session_start_time=" << std::fixed << std::setprecision(6)
                       << ros::Time::now().toSec() << " log_version=2\n"
                       << "===================================================\n";
      update_log_file_.flush();
    }
    else
    {
      ROS_WARN("[UWB] Failed to open UWB update log file: %s", update_log_path.c_str());
    }
    if (!replay_source)
    {
      raw_log_file_.open(raw_log_path, std::ios::out | std::ios::app);
      if (raw_log_file_.is_open())
      {
        raw_log_file_ << "# stamp raw_line parsed_ranges(raw,bias,corrected)_m\n";
        raw_log_file_.flush();
      }
      else
      {
        ROS_WARN("[UWB] Failed to open UWB raw replay log file: %s", raw_log_path.c_str());
      }
    }
  }

  logAnchorConfiguration();
  if (!anchor_frame_align_en_ && !baseline_anchor_init_en_ && !anchors_.empty())
  {
    logFinalAnchorLayout("manual_configuration", "camera_init");
  }
  if (!tag_offset_estimate_en_)
  {
    logFinalTagOffset("manual_configuration");
  }

  if (replay_source)
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
        if (two_anchor_baseline_mode_ && !configured_anchors_.empty())
        {
          ROS_INFO("[UWB] YAML anchors provide ids and a baseline template in an external frame; trajectory-aligned baseline initialization will produce final camera_init coordinates.");
          logEvent(ros::Time::now().toSec(), "INFO",
                   "UWB_ANCHOR_SOURCE anchor_position_source=trajectory_aligned_baseline "
                   "yaml_position_used_as_template=1 yaml_position_used_as_final_world_coordinate=0 "
                   "source=replay");
        }
        else
        {
          ROS_WARN("[UWB] No world-frame anchors were configured. Baseline anchor initialization will use start/end ids and known distance before EKF updates start.");
          logEvent(ros::Time::now().toSec(), "WARN",
                   "NO_WORLD_FRAME_ANCHORS baseline_anchor_init source=replay");
        }
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
  ROS_INFO("[UWB] Serial reader started: port=%s baud=%d DTR=%d RTS=%d raw_log=%s update_log=%s",
           serial_port_.c_str(), baudrate_, static_cast<int>(dtr_high_), static_cast<int>(rts_high_),
           raw_log_path.c_str(), update_log_path.c_str());
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
      if (two_anchor_baseline_mode_ && !configured_anchors_.empty())
      {
        ROS_INFO("[UWB] YAML anchors provide ids and a baseline template in an external frame; trajectory-aligned baseline initialization will produce final camera_init coordinates.");
        logEvent(ros::Time::now().toSec(), "INFO",
                 "UWB_ANCHOR_SOURCE anchor_position_source=trajectory_aligned_baseline "
                 "yaml_position_used_as_template=1 yaml_position_used_as_final_world_coordinate=0 "
                 "source=serial");
      }
      else
      {
        ROS_WARN("[UWB] No world-frame anchors were configured. Baseline anchor initialization will use start/end ids and known distance before EKF updates start.");
        logEvent(ros::Time::now().toSec(), "WARN",
                 "NO_WORLD_FRAME_ANCHORS baseline_anchor_init source=serial");
      }
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
    if (raw_log_file_.is_open())
    {
      raw_log_file_.flush();
      raw_log_file_.close();
    }
    if (update_log_file_.is_open())
    {
      update_log_file_.flush();
      update_log_file_.close();
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
  nh.param<std::string>("uwb/update_log_filename", update_log_filename_, "uwb_updates.txt");
  nh.param<int>("uwb/log_flush_stride", log_flush_stride_, 1);
  nh.param<bool>("uwb/summary_log_en", summary_log_en_, true);
  nh.param<bool>("uwb/debug_log_en", debug_log_en_, true);
  nh.param<bool>("uwb/range_debug_log_en", range_debug_log_en_, true);
  nh.param<bool>("uwb/summary_to_console", summary_to_console_, true);
  nh.param<bool>("uwb/debug_to_console", debug_to_console_, false);
  nh.param<bool>("uwb/summary_to_file", summary_to_file_, true);
  nh.param<bool>("uwb/debug_to_file", debug_to_file_, true);
  nh.param<int>("uwb/statistics_log_interval", statistics_log_interval_, 20);
  nh.param<double>("uwb/update_epsilon", update_epsilon_, 1e-8);
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
  int legacy_min_anchors = min_anchors_for_update_;
  int legacy_min_update_anchors = min_anchors_for_update_;
  const bool has_legacy_min_anchors = nh.getParam("uwb/min_anchors", legacy_min_anchors);
  const bool has_legacy_min_update_anchors = nh.getParam("uwb/min_update_anchors", legacy_min_update_anchors);
  const bool has_canonical_min_anchors =
      nh.getParam("uwb/min_anchors_for_update", min_anchors_for_update_);
  if (has_canonical_min_anchors && (has_legacy_min_anchors || has_legacy_min_update_anchors))
  {
    ROS_WARN("[UWB] Deprecated uwb/min_anchors or uwb/min_update_anchors also present. Use canonical uwb/min_anchors_for_update=%d.",
             min_anchors_for_update_);
  }
  else if (has_legacy_min_update_anchors)
  {
    min_anchors_for_update_ = legacy_min_update_anchors;
    ROS_WARN("[UWB] uwb/min_update_anchors is deprecated. Please rename it to uwb/min_anchors_for_update.");
  }
  else if (has_legacy_min_anchors)
  {
    min_anchors_for_update_ = legacy_min_anchors;
    ROS_WARN("[UWB] uwb/min_anchors is deprecated. Please rename it to uwb/min_anchors_for_update.");
  }
  min_update_anchors_ = min_anchors_for_update_;
  nh.param<int>("uwb/prefer_anchors", prefer_anchors_, 3);
  nh.param<double>("uwb/sigma", range_noise_m_, 0.10);
  nh.param<double>("uwb/range_noise_m", range_noise_m_, range_noise_m_);
  nh.param<double>("uwb/position_cov_floor_m", position_cov_floor_m_, 0.0);
  nh.param<double>("uwb/max_residual_m", max_residual_m_, 6.0);
  nh.param<double>("uwb/max_residual_rms", max_residual_rms_, 0.50);
  nh.param<double>("uwb/max_xy_correction_normal", max_xy_correction_normal_, 0.50);
  nh.param<double>("uwb/normal_update_max_xy_raw", normal_update_max_xy_raw_, 0.80);
  nh.param<double>("uwb/max_update_step_xy", max_update_step_xy_, 0.05);
  nh.param<double>("uwb/two_anchor_sigma_scale", two_anchor_sigma_scale_, 5.0);
  nh.param<std::string>("uwb/two_anchor_update_mode", two_anchor_update_mode_, "baseline_1d_direct");
  nh.param<std::string>("uwb/two_anchor_policy_when_total_anchors_gt2",
                        two_anchor_policy_when_total_anchors_gt2_, "dry_run");
  nh.param<double>("uwb/baseline_1d_direct_alpha", baseline_1d_direct_alpha_, 0.05);
  nh.param<double>("uwb/baseline_1d_direct_max_step_m", baseline_1d_direct_max_step_m_, 0.03);
  nh.param<double>("uwb/two_anchor_baseline_direct_alpha",
                   baseline_1d_direct_alpha_, baseline_1d_direct_alpha_);
  nh.param<double>("uwb/two_anchor_baseline_direct_max_step_m",
                   baseline_1d_direct_max_step_m_, baseline_1d_direct_max_step_m_);
  nh.param<double>("uwb/two_anchor_alpha", baseline_1d_direct_alpha_, baseline_1d_direct_alpha_);
  nh.param<double>("uwb/two_anchor_normal_max_step", two_anchor_normal_max_step_m_, 0.05);
  nh.param<double>("uwb/two_anchor_degraded_max_step", two_anchor_degraded_max_step_m_, 0.10);
  nh.param<double>("uwb/two_anchor_strong_degraded_max_step", two_anchor_strong_degraded_max_step_m_, 0.15);
  nh.param<double>("uwb/two_anchor_hard_max_step", two_anchor_hard_max_step_m_, 0.20);
  nh.param<double>("uwb/two_anchor_max_residual", two_anchor_max_residual_, 2.0);
  nh.param<double>("uwb/two_anchor_baseline_consistency_threshold_m",
                   two_anchor_baseline_consistency_threshold_m_, 2.0);
  nh.param<double>("uwb/baseline_consistency_threshold",
                   two_anchor_baseline_consistency_threshold_m_, two_anchor_baseline_consistency_threshold_m_);
  nh.param<double>("uwb/two_anchor_max_residual_rms", two_anchor_max_residual_rms_, 0.8);
  nh.param<double>("uwb/two_anchor_max_abs_residual", two_anchor_max_abs_residual_, 1.5);
  nh.param<bool>("uwb/single_anchor_corridor_1d_en", single_anchor_corridor_1d_en_, true);
  nh.param<bool>("uwb/single_anchor_only_when_total_anchors_eq_2",
                 single_anchor_only_when_total_anchors_eq_2_, true);
  nh.param<bool>("uwb/single_anchor_requires_baseline_initialized",
                 single_anchor_requires_baseline_initialized_, true);
  nh.param<double>("uwb/single_anchor_alpha", single_anchor_alpha_, 0.05);
  nh.param<double>("uwb/single_anchor_normal_max_step", single_anchor_normal_max_step_m_, 0.05);
  nh.param<double>("uwb/single_anchor_degraded_max_step", single_anchor_degraded_max_step_m_, 0.10);
  nh.param<double>("uwb/single_anchor_strong_degraded_max_step",
                   single_anchor_strong_degraded_max_step_m_, 0.15);
  nh.param<double>("uwb/single_anchor_hard_max_step", single_anchor_hard_max_step_m_, 0.20);
  nh.param<double>("uwb/single_anchor_max_residual", single_anchor_max_residual_, 2.0);
  nh.param<int>("uwb/single_anchor_confirm_count", single_anchor_confirm_count_required_, 1);
  nh.param<double>("uwb/single_anchor_min_range", single_anchor_min_range_m_, 1.0);
  nh.param<double>("uwb/single_anchor_max_range", single_anchor_max_range_m_, 60.0);
  nh.param<double>("uwb/single_anchor_branch_margin", single_anchor_branch_margin_m_, 0.3);
  nh.param<double>("uwb/single_anchor_near_anchor_disable_dist",
                   single_anchor_near_anchor_disable_dist_m_, 0.3);
  nh.param<double>("uwb/single_anchor_range_jump_threshold", single_anchor_range_jump_threshold_m_, 2.0);
  nh.param<double>("uwb/single_anchor_residual_jump_threshold",
                   single_anchor_residual_jump_threshold_m_, 1.0);
  nh.param<double>("uwb/single_anchor_speed_threshold", single_anchor_speed_threshold_mps_, 2.0);
  nh.param<double>("uwb/corridor_direction_max_angle_deg", corridor_direction_max_angle_deg_, 45.0);
  nh.param<double>("uwb/min_motion_for_direction_check", min_motion_for_direction_check_m_, 0.3);
  nh.param<int>("uwb/direction_check_window_frames", direction_check_window_frames_, 10);
  nh.param<bool>("uwb/disable_single_anchor_on_turn", disable_single_anchor_on_turn_, true);
  nh.param<bool>("uwb/enable_corridor_segments", enable_corridor_segments_, false);
  nh.param<bool>("uwb/degraded_mode_en", degraded_mode_en_, true);
  nh.param<int>("uwb/degraded_confirm_count", degraded_confirm_count_, 3);
  nh.param<int>("uwb/strong_degraded_confirm_count", strong_degraded_confirm_count_, 5);
  nh.param<double>("uwb/multi_anchor_max_residual_rms", multi_anchor_max_residual_rms_, 1.0);
  nh.param<double>("uwb/multi_anchor_max_abs_residual", multi_anchor_max_abs_residual_, 1.5);
  nh.param<double>("uwb/max_time_diff", max_time_diff_s_, 0.05);
  nh.param<double>("uwb/max_time_diff_s", max_time_diff_s_, max_time_diff_s_);
  nh.param<double>("uwb/limited_update_max_residual_rms", limited_update_max_residual_rms_, 2.0);
  nh.param<double>("uwb/limited_update_max_abs_residual", limited_update_max_abs_residual_, 3.0);
  nh.param<double>("uwb/limited_update_max_xy_raw", limited_update_max_xy_raw_, 2.0);
  nh.param<double>("uwb/limited_update_max_time_diff_s", limited_update_max_time_diff_s_, replay_match_threshold_s_);
  nh.param<int>("uwb/limited_update_require_consecutive_good",
                limited_update_require_consecutive_good_, 2);
  nh.param<double>("uwb/relocalization_candidate_min_xy_raw",
                   relocalization_candidate_min_xy_raw_, 1.5);
  nh.param<double>("uwb/hard_reject_xy_raw", hard_reject_xy_raw_, 3.0);
  nh.param<bool>("uwb/uwb_debug_force_limited_update", uwb_debug_force_limited_update_, false);
  nh.param<bool>("uwb/relocalization_en", relocalization_en_, false);
  nh.param<double>("uwb/relocalization_threshold", relocalization_threshold_m_, 1.5);
  nh.param<int>("uwb/relocalization_confirm_count", relocalization_confirm_count_, 5);
  nh.param<double>("uwb/uwb_only_max_residual_rms", uwb_only_max_residual_rms_, 0.5);
  nh.param<double>("uwb/uwb_only_max_abs_residual", uwb_only_max_abs_residual_, 1.0);
  nh.param<double>("uwb/uwb_position_jump_threshold", uwb_position_jump_threshold_m_, 1.0);
  nh.param<double>("uwb/uwb_speed_threshold", uwb_speed_threshold_mps_, 2.0);
  nh.param<double>("uwb/anchor_geometry_min_score", anchor_geometry_min_score_, 1e-3);
  nh.param<int>("uwb/require_consecutive_good_updates", require_consecutive_good_updates_, 3);
  nh.param<double>("uwb/good_residual_rms", good_residual_rms_, 0.30);
  nh.param<bool>("uwb/suspect_hold_en", suspect_hold_en_, false);
  nh.param<bool>("uwb/lost_hold_en", lost_hold_en_, false);
  nh.param<double>("uwb/large_correction_warn_threshold", large_correction_warn_threshold_, 0.50);
  nh.param<double>("uwb/large_correction_reject_threshold", large_correction_reject_threshold_, 3.0);
  nh.param<std::string>("uwb/anchor_file", anchor_file_, "");
  nh.param<double>("uwb/position_cov_floor_degraded_m", position_cov_floor_degraded_m_, 0.0);
  nh.param<bool>("uwb/position_cov_floor_degraded_only", position_cov_floor_degraded_only_, true);
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
  nh.param<double>("uwb/tag_offset_convergence_step_m", tag_offset_convergence_step_m_, 1e-4);
  nh.param<int>("uwb/tag_offset_convergence_count", tag_offset_convergence_count_, 20);
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
  statistics_log_interval_ = std::max(1, statistics_log_interval_);
  update_epsilon_ = std::max(0.0, update_epsilon_);
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
  min_anchors_for_update_ = std::max(1, min_anchors_for_update_);
  min_update_anchors_ = min_anchors_for_update_;
  prefer_anchors_ = std::max(0, prefer_anchors_);
  range_noise_m_ = std::max(1e-3, range_noise_m_);
  max_residual_rms_ = std::max(0.0, max_residual_rms_);
  max_xy_correction_normal_ = std::max(0.0, max_xy_correction_normal_);
  normal_update_max_xy_raw_ = std::max(0.0, normal_update_max_xy_raw_);
  max_update_step_xy_ = std::max(0.0, max_update_step_xy_);
  two_anchor_sigma_scale_ = std::max(1.0, two_anchor_sigma_scale_);
  two_anchor_update_mode_ = toLower(two_anchor_update_mode_);
  two_anchor_policy_when_total_anchors_gt2_ = toLower(two_anchor_policy_when_total_anchors_gt2_);
  std::replace(two_anchor_policy_when_total_anchors_gt2_.begin(),
               two_anchor_policy_when_total_anchors_gt2_.end(), '-', '_');
  if (two_anchor_policy_when_total_anchors_gt2_ != "dry_run" &&
      two_anchor_policy_when_total_anchors_gt2_ != "weak_xy" &&
      two_anchor_policy_when_total_anchors_gt2_ != "baseline_1d_only_if_pair_matches_corridor" &&
      two_anchor_policy_when_total_anchors_gt2_ != "disable")
  {
    ROS_WARN("[UWB] Unknown two_anchor_policy_when_total_anchors_gt2=%s. Use dry_run.",
             two_anchor_policy_when_total_anchors_gt2_.c_str());
    two_anchor_policy_when_total_anchors_gt2_ = "dry_run";
  }
  if (two_anchor_update_mode_ == "baseline_1d_direct_update")
  {
    two_anchor_update_mode_ = "baseline_1d_direct";
  }
  if (two_anchor_update_mode_ != "dry_run" &&
      two_anchor_update_mode_ != "baseline_1d" &&
      two_anchor_update_mode_ != "baseline_1d_direct" &&
      two_anchor_update_mode_ != "weak_xy")
  {
    ROS_WARN("[UWB] Unknown two_anchor_update_mode=%s. Use baseline_1d_direct.", two_anchor_update_mode_.c_str());
    two_anchor_update_mode_ = "baseline_1d_direct";
  }
  baseline_1d_direct_alpha_ = std::max(0.0, baseline_1d_direct_alpha_);
  baseline_1d_direct_max_step_m_ = std::max(0.0, baseline_1d_direct_max_step_m_);
  two_anchor_normal_max_step_m_ = std::max(0.0, two_anchor_normal_max_step_m_);
  two_anchor_degraded_max_step_m_ = std::max(two_anchor_normal_max_step_m_, two_anchor_degraded_max_step_m_);
  two_anchor_strong_degraded_max_step_m_ = std::max(two_anchor_degraded_max_step_m_, two_anchor_strong_degraded_max_step_m_);
  two_anchor_hard_max_step_m_ = std::max(two_anchor_strong_degraded_max_step_m_, two_anchor_hard_max_step_m_);
  two_anchor_max_residual_ = std::max(0.0, two_anchor_max_residual_);
  two_anchor_baseline_consistency_threshold_m_ = std::max(0.0, two_anchor_baseline_consistency_threshold_m_);
  two_anchor_max_residual_rms_ = std::max(0.0, two_anchor_max_residual_rms_);
  two_anchor_max_abs_residual_ = std::max(0.0, two_anchor_max_abs_residual_);
  single_anchor_alpha_ = std::max(0.0, single_anchor_alpha_);
  single_anchor_normal_max_step_m_ = std::max(0.0, single_anchor_normal_max_step_m_);
  single_anchor_degraded_max_step_m_ = std::max(single_anchor_normal_max_step_m_, single_anchor_degraded_max_step_m_);
  single_anchor_strong_degraded_max_step_m_ =
      std::max(single_anchor_degraded_max_step_m_, single_anchor_strong_degraded_max_step_m_);
  single_anchor_hard_max_step_m_ = std::max(single_anchor_strong_degraded_max_step_m_, single_anchor_hard_max_step_m_);
  single_anchor_max_residual_ = std::max(0.0, single_anchor_max_residual_);
  single_anchor_confirm_count_required_ = std::max(1, single_anchor_confirm_count_required_);
  single_anchor_min_range_m_ = std::max(0.0, single_anchor_min_range_m_);
  single_anchor_max_range_m_ = std::max(single_anchor_min_range_m_, single_anchor_max_range_m_);
  single_anchor_branch_margin_m_ = std::max(0.0, single_anchor_branch_margin_m_);
  single_anchor_near_anchor_disable_dist_m_ = std::max(0.0, single_anchor_near_anchor_disable_dist_m_);
  single_anchor_range_jump_threshold_m_ = std::max(0.0, single_anchor_range_jump_threshold_m_);
  single_anchor_residual_jump_threshold_m_ = std::max(0.0, single_anchor_residual_jump_threshold_m_);
  single_anchor_speed_threshold_mps_ = std::max(0.0, single_anchor_speed_threshold_mps_);
  corridor_direction_max_angle_deg_ = std::max(0.0, std::min(90.0, corridor_direction_max_angle_deg_));
  min_motion_for_direction_check_m_ = std::max(0.0, min_motion_for_direction_check_m_);
  direction_check_window_frames_ = std::max(2, direction_check_window_frames_);
  degraded_confirm_count_ = std::max(1, degraded_confirm_count_);
  strong_degraded_confirm_count_ = std::max(degraded_confirm_count_, strong_degraded_confirm_count_);
  multi_anchor_max_residual_rms_ = std::max(0.0, multi_anchor_max_residual_rms_);
  multi_anchor_max_abs_residual_ = std::max(0.0, multi_anchor_max_abs_residual_);
  max_time_diff_s_ = std::max(0.0, max_time_diff_s_);
  limited_update_max_residual_rms_ = std::max(0.0, limited_update_max_residual_rms_);
  limited_update_max_abs_residual_ = std::max(0.0, limited_update_max_abs_residual_);
  limited_update_max_xy_raw_ = std::max(0.0, limited_update_max_xy_raw_);
  limited_update_max_time_diff_s_ = std::max(0.0, limited_update_max_time_diff_s_);
  limited_update_require_consecutive_good_ = std::max(0, limited_update_require_consecutive_good_);
  relocalization_candidate_min_xy_raw_ = std::max(0.0, relocalization_candidate_min_xy_raw_);
  hard_reject_xy_raw_ = std::max(0.0, hard_reject_xy_raw_);
  relocalization_threshold_m_ = std::max(0.0, relocalization_threshold_m_);
  relocalization_confirm_count_ = std::max(1, relocalization_confirm_count_);
  uwb_only_max_residual_rms_ = std::max(0.0, uwb_only_max_residual_rms_);
  uwb_only_max_abs_residual_ = std::max(0.0, uwb_only_max_abs_residual_);
  uwb_position_jump_threshold_m_ = std::max(0.0, uwb_position_jump_threshold_m_);
  uwb_speed_threshold_mps_ = std::max(0.0, uwb_speed_threshold_mps_);
  anchor_geometry_min_score_ = std::max(0.0, anchor_geometry_min_score_);
  require_consecutive_good_updates_ = std::max(0, require_consecutive_good_updates_);
  good_residual_rms_ = std::max(0.0, good_residual_rms_);
  large_correction_warn_threshold_ = std::max(max_xy_correction_normal_, large_correction_warn_threshold_);
  large_correction_reject_threshold_ = std::max(large_correction_warn_threshold_, large_correction_reject_threshold_);
  position_cov_floor_m_ = std::max(0.0, position_cov_floor_m_);
  position_cov_floor_degraded_m_ = std::max(0.0, position_cov_floor_degraded_m_);
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
  tag_offset_convergence_step_m_ = std::max(0.0, tag_offset_convergence_step_m_);
  tag_offset_convergence_count_ = std::max(1, tag_offset_convergence_count_);
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

      if (xmlRpcHasMember(anchor_param, "role") &&
          !xmlRpcToString(anchor_param["role"], anchor.role))
      {
        ROS_WARN("[UWB] Anchor %d role must be a string. Use the role inferred from its id.", anchor.id);
        anchor.role.clear();
      }

      if (xmlRpcHasMember(anchor_param, "range_bias_m"))
      {
        if (!xmlRpcToDouble(anchor_param["range_bias_m"], anchor.range_bias_m) ||
            !std::isfinite(anchor.range_bias_m))
        {
          ROS_WARN("[UWB] Anchor %d has invalid range_bias_m. Use 0.0 m.", anchor.id);
          anchor.range_bias_m = 0.0;
        }
      }
      if (anchor.range_bias_m < 0.0)
      {
        ROS_WARN("[UWB] Anchor %d uses negative range_bias_m=%.6f m; corrected_range will be larger than raw_range.",
                 anchor.id, anchor.range_bias_m);
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
  if (two_anchor_baseline_mode_)
  {
    ROS_INFO("[UWB] anchor_position_source=trajectory_aligned_baseline yaml_position_used_as_template=1 yaml_position_used_as_final_world_coordinate=0");
  }
  ROS_INFO("[UWB] effective_config total_configured_anchors=%zu canonical_min_anchors_for_update=%d prefer_anchors=%d mode=%s anchor_frame_align_en=%d two_anchor_update_mode=%s two_anchor_policy_when_total_anchors_gt2=%s single_anchor_corridor_1d_en=%d single_anchor_only_when_total_anchors_eq_2=%d replay_match_threshold_s=%.3f replay_start_offset_s=%.3f",
           enabled_configured_anchor_ids.size(),
           min_anchors_for_update_,
           prefer_anchors_,
           mode_.c_str(),
           static_cast<int>(anchor_frame_align_en_),
           two_anchor_update_mode_.c_str(),
           two_anchor_policy_when_total_anchors_gt2_.c_str(),
           static_cast<int>(single_anchor_corridor_1d_en_),
           static_cast<int>(single_anchor_only_when_total_anchors_eq_2_),
           replay_match_threshold_s_,
           replay_start_offset_s_);
  ROS_INFO("[UWB] range_model=%s update_xy_only=%d update_z=%d update_orientation=%d canonical_min_anchors_for_update=%d prefer_anchors=%d sigma=%.3f",
           use_3d_range_model_ ? "3d" : "legacy_xy",
           static_cast<int>(update_xy_only_),
           static_cast<int>(update_z_),
           static_cast<int>(update_orientation_),
           min_anchors_for_update_,
           prefer_anchors_,
           range_noise_m_);
  ROS_INFO("[UWB] update_strategy residual_debug_only=%d max_update_step_xy=%.3f min_anchors_for_update=%d two_anchor_mode=%s two_anchor_sigma_scale=%.3f baseline_direct_alpha=%.3f baseline_direct_max_step=%.3f require_good=%d good_rms=%.3f limited_good=%d normal_xy=%.3f limited_xy=%.3f relocalize_xy=%.3f hard_reject_xy=%.3f suspect_hold=%d lost_hold=%d",
           static_cast<int>(residual_debug_only_),
           max_update_step_xy_,
           min_anchors_for_update_,
           two_anchor_update_mode_.c_str(),
           two_anchor_sigma_scale_,
           baseline_1d_direct_alpha_,
           baseline_1d_direct_max_step_m_,
           require_consecutive_good_updates_,
           good_residual_rms_,
           limited_update_require_consecutive_good_,
           normal_update_max_xy_raw_,
           limited_update_max_xy_raw_,
           relocalization_candidate_min_xy_raw_,
           hard_reject_xy_raw_,
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
  replay_consumed_measurement_count_ = 0;
  replay_stale_measurement_count_ = 0;
  repeated_range_states_.clear();

  uint64_t total_parsed = 0;
  uint64_t invalid_zero_filtered = 0;
  uint64_t invalid_nonfinite_filtered = 0;
  uint64_t invalid_corrected_range_filtered = 0;
  uint64_t range_limit_filtered = 0;
  uint64_t invalid_other_filtered = 0;
  uint64_t duplicate_filtered = 0;
  uint64_t unknown_anchor_filtered = 0;
  size_t source_line = 0;
  std::vector<UwbRangeMeasurement> replay_candidates;
  std::string line;
  while (std::getline(replay_file, line))
  {
    source_line++;
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

    // ponytail: replay de-duplication is timestamp-aware below; the range-only stale filter stays serial-only.
    auto measurements = parseLine(raw_line, stamp);
    total_parsed += measurements.size();
    for (auto &measurement : measurements)
    {
      measurement.source_line = source_line;
      measurement.stamp = stamp;
      if (!std::isfinite(measurement.raw_range_m))
      {
        invalid_nonfinite_filtered++;
        continue;
      }
      if (measurement.raw_range_m <= 0.0)
      {
        invalid_zero_filtered++;
        continue;
      }
      if (!configured_anchors_.empty() &&
          configured_anchors_.find(measurement.anchor_id) == configured_anchors_.end())
      {
        unknown_anchor_filtered++;
        continue;
      }
      if (!measurement.range_valid)
      {
        if (measurement.range_reject_reason == UwbRejectReason::INVALID_CORRECTED_RANGE)
          invalid_corrected_range_filtered++;
        else if (measurement.range_reject_reason == UwbRejectReason::RANGE_LIMIT)
          range_limit_filtered++;
        else
          invalid_other_filtered++;
        continue;
      }
      replay_candidates.push_back(measurement);
    }
  }

  std::sort(replay_candidates.begin(), replay_candidates.end(),
            [](const UwbRangeMeasurement &a, const UwbRangeMeasurement &b) {
              return a.stamp < b.stamp;
            });

  std::set<std::tuple<int, int64_t, int64_t>> seen_measurement_keys;
  std::set<std::pair<int, int64_t>> seen_diag_keys;
  constexpr double kCrossFormatDuplicateWindowS = 0.05;
  constexpr size_t kDuplicateDebugLimit = 5;
  size_t duplicate_debug_count = 0;
  for (const auto &measurement : replay_candidates)
  {
    const int64_t timestamp_us =
        static_cast<int64_t>(std::llround(measurement.stamp * 1e6));
    const int64_t range_mm =
        static_cast<int64_t>(std::llround(measurement.raw_range_m * 1000.0));
    const auto key = std::make_tuple(measurement.anchor_id, timestamp_us, range_mm);
    if (measurement.diag >= 0 &&
        !seen_diag_keys.insert({measurement.anchor_id, measurement.diag}).second)
    {
      duplicate_filtered++;
      continue;
    }
    bool cross_format_duplicate = false;
    const UwbRangeMeasurement *kept_distance_measurement = nullptr;
    if (measurement.source_format == "uwbdbg")
    {
      // ponytail: current replay files are small; upgrade this bounded scan to an indexed key if they grow large.
      for (const auto &candidate : replay_candidates)
      {
        if (!isUwbReplayCrossFormatDuplicate(
                measurement, candidate, kCrossFormatDuplicateWindowS))
          continue;
        cross_format_duplicate = true;
        kept_distance_measurement = &candidate;
        break;
      }
    }
    if (cross_format_duplicate)
    {
      duplicate_filtered++;
      if (debug_log_en_ && duplicate_debug_count < kDuplicateDebugLimit)
      {
        std::ostringstream oss;
        oss << "[UWB_REPLAY_DUPLICATE] anchor=" << measurement.anchor_id
            << " diag=" << measurement.diag
            << " timestamp_us=" << timestamp_us
            << " range_mm=" << range_mm
            << " filtered_source=" << measurement.source_format
            << " filtered_source_line=" << measurement.source_line
            << " kept_source=distance"
            << " kept_source_line="
            << (kept_distance_measurement != nullptr ?
                kept_distance_measurement->source_line : 0);
        emitUwbLine(measurement.stamp, "INFO", oss.str(),
                    debug_to_console_, debug_to_file_);
        duplicate_debug_count++;
      }
      continue;
    }
    if (!seen_measurement_keys.insert(key).second)
    {
      duplicate_filtered++;
      continue;
    }
    replay_measurements_.push_back(measurement);
  }

  for (size_t i = 0; i < replay_measurements_.size(); ++i)
    replay_measurements_[i].measurement_uid = static_cast<uint64_t>(i + 1);

  {
    std::ostringstream oss;
    oss << "[UWB_REPLAY_LOAD_STATS]"
        << " total_parsed=" << total_parsed
        << " valid_loaded=" << replay_measurements_.size()
        << " invalid_zero_filtered=" << invalid_zero_filtered
        << " invalid_nonfinite_filtered=" << invalid_nonfinite_filtered
        << " invalid_corrected_range_filtered=" << invalid_corrected_range_filtered
        << " range_limit_filtered=" << range_limit_filtered
        << " invalid_other_filtered=" << invalid_other_filtered
        << " duplicate_filtered=" << duplicate_filtered
        << " unknown_anchor_filtered=" << unknown_anchor_filtered;
    emitUwbLine(ros::Time::now().toSec(), "INFO", oss.str(), true, true);
  }

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
          << " measurement_uid=" << measurement.measurement_uid
          << " measurement_stamp=" << measurement.stamp
          << " file_start_stamp=" << replay_file_start_stamp_
          << " measurement_relative=" << (measurement.stamp - replay_file_start_stamp_)
          << " start_offset=" << replay_start_offset_s_
          << " uwb_relative_time=" << uwb_relative_time;
      logEventThrottled(current_lidar_stamp, "skip_replay_before_zero", 3.0, "INFO", oss.str());
      replay_stale_measurement_count_++;
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
            << " measurement_uid=" << measurement.measurement_uid
            << " slam_relative_time=" << slam_relative_time
            << " uwb_relative_time=" << uwb_relative_time
            << " dt=" << dt
            << " abs_dt=" << std::fabs(dt)
            << " match_threshold=" << match_threshold;
        logEventThrottled(current_lidar_stamp, "drop_stale_replay", 3.0, "WARN", oss.str());
      }
      replay_stale_measurement_count_++;
      replay_index_++;
      continue;
    }
    UwbRangeMeasurement matched_measurement = measurement;
    matched_measurement.time_diff_s = dt;
    measurements.push_back(matched_measurement);
    replay_consumed_measurement_count_++;
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

  auto appendMeasurement = [&](int anchor_id, double raw_range,
                               const std::string &source_format, int64_t diag)
  {
    UwbRangeMeasurement measurement;
    measurement.anchor_id = anchor_id;
    measurement.raw_range_m = raw_range * range_scale_;
    measurement.range_valid =
        correctUwbRange(anchor_id, measurement.raw_range_m, measurement.range_m,
                        measurement.range_bias_m, measurement.range_reject_reason);
    measurement.source_format = source_format;
    measurement.diag = diag;
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
      appendMeasurement(std::stoi(distance_match[1].str()), std::stod(distance_match[2].str()),
                        "distance", -1);
    }
    return measurements;
  }
  if (parser_mode_ == "distance") return measurements;

  static const std::regex target_regex(R"(\btarget\s*=\s*([-+]?\d+))", std::regex::icase);
  static const std::regex ok_regex(R"(\bok\s*=\s*([-+]?\d+))", std::regex::icase);
  static const std::regex diag_regex(R"(\bdiag\s*=\s*([-+]?\d+))", std::regex::icase);
  static const std::regex dist_regex(R"(\bdist\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))", std::regex::icase);
  const bool parse_debug_distance = parser_mode_ == "uwb" || parser_mode_ == "auto";
  if (parse_debug_distance &&
      (line.find("[UWBDBG]") != std::string::npos || line.find("dist=") != std::string::npos))
  {
    int target_id = -1;
    int ok = 0;
    int diag = -1;
    double dist_m = 0.0;
    if (regexFindInt(line, target_regex, target_id) &&
        regexFindInt(line, ok_regex, ok) &&
        regexFindDouble(line, dist_regex, dist_m) &&
        ok == 1 &&
        !lineContainsErrorStatus(line))
    {
      regexFindInt(line, diag_regex, diag);
      appendMeasurement(target_id, dist_m,
                        line.find("[UWBDBG]") != std::string::npos ? "uwbdbg" : "debug_distance",
                        diag);
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
      appendMeasurement(static_cast<int>(std::llround(values[i])), values[i + 1], "pairs", -1);
    }
    return measurements;
  }

  for (size_t i = 0; i < values.size(); ++i)
  {
    const int anchor_id = (i < anchor_order_.size()) ? anchor_order_[i] : static_cast<int>(i);
    appendMeasurement(anchor_id, values[i], "values", -1);
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
    if (!measurement.range_valid)
    {
      filtered.push_back(measurement);
      continue;
    }
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
          << " raw_range=" << measurement.raw_range_m
          << " range_bias=" << measurement.range_bias_m
          << " corrected_range=" << measurement.range_m
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
  if (!raw_log_file_.is_open()) return;

  raw_log_file_ << std::fixed << std::setprecision(6) << stamp << " raw=\"" << line << "\" parsed=";
  if (measurements.empty())
  {
    raw_log_file_ << "none";
  }
  else
  {
    for (const auto &measurement : measurements)
    {
      raw_log_file_ << measurement.anchor_id
                    << ":raw=" << std::setprecision(4) << measurement.raw_range_m << "m"
                    << ",bias=" << measurement.range_bias_m << "m"
                    << ",corrected=" << measurement.range_m << "m"
                    << ",valid=" << static_cast<int>(measurement.range_valid) << " ";
    }
  }
  raw_log_file_ << '\n';
  raw_log_pending_lines_++;
  if (raw_log_pending_lines_ >= log_flush_stride_)
  {
    raw_log_file_.flush();
    raw_log_pending_lines_ = 0;
  }
}

void UwbManager::logEvent(double stamp, const std::string &level, const std::string &message)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!update_log_file_.is_open()) return;

  update_log_file_ << std::fixed << std::setprecision(6)
                   << stamp << " " << level << " " << message << '\n';
  update_log_file_.flush();
}

void UwbManager::logEventThrottled(double stamp, const std::string &key, double period_s,
                                   const std::string &level, const std::string &message)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!update_log_file_.is_open()) return;

  const auto it = event_log_last_stamp_.find(key);
  if (period_s > 0.0 && it != event_log_last_stamp_.end() && stamp - it->second < period_s)
  {
    return;
  }
  event_log_last_stamp_[key] = stamp;

  update_log_file_ << std::fixed << std::setprecision(6)
                   << stamp << " " << level << " " << message << '\n';
  update_log_file_.flush();
}

void UwbManager::emitUwbLine(double stamp, const std::string &level, const std::string &line,
                             bool console, bool file)
{
  if (console)
  {
    if (level == "WARN" || level == "ERROR") ROS_WARN_STREAM(line);
    else ROS_INFO_STREAM(line);
  }
  if (file) logEvent(stamp, level, line);
}

void UwbManager::logAnchorConfiguration()
{
  const double stamp = ros::Time::now().toSec();
  std::set<int> logged_ids;
  for (int id : anchor_order_)
  {
    const auto it = configured_anchors_.find(id);
    if (it == configured_anchors_.end() || !logged_ids.insert(id).second) continue;
    const auto &anchor = it->second;
    const std::string role = !anchor.role.empty() ? anchor.role :
                             (id == baseline_anchor_start_id_ ? "entry" :
                              (id == baseline_anchor_end_id_ ? "exit" : "anchor"));
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "[UWB_ANCHOR_CONFIG] anchor=" << id
        << " role=" << role
        << " position=(" << anchor.position_w.x() << "," << anchor.position_w.y() << ","
        << anchor.position_w.z() << ")"
        << " range_bias_m=" << anchor.range_bias_m;
    emitUwbLine(stamp, "INFO", oss.str(), true, true);
  }
  emitUwbLine(stamp, "INFO",
              "measurement_correction_formula=corrected_range=raw_range-range_bias_m",
              true, true);
}

void UwbManager::logFinalAnchorLayout(const std::string &source, const std::string &frame_name,
                                      UwbUpdateReport *report)
{
  std::vector<int> ordered_ids;
  std::set<int> seen;
  for (int id : anchor_order_)
  {
    const auto it = anchors_.find(id);
    if (it != anchors_.end() && it->second.enabled && seen.insert(id).second) ordered_ids.push_back(id);
  }
  for (const auto &item : anchors_)
  {
    if (item.second.enabled && seen.insert(item.first).second) ordered_ids.push_back(item.first);
  }
  if (ordered_ids.empty()) return;

  std::ostringstream signature;
  signature << std::fixed << std::setprecision(6);
  for (int id : ordered_ids)
  {
    const auto &anchor = anchors_.at(id);
    signature << id << ":" << anchor.enabled << ":" << anchor.estimated << ":"
              << anchor.role << ":" << anchor.range_bias_m << ":"
              << anchor.position_w.transpose() << ";";
  }
  if (final_anchor_layout_logged_ && signature.str() == final_anchor_layout_signature_) return;
  final_anchor_layout_signature_ = signature.str();
  final_anchor_layout_logged_ = true;
  anchor_layout_version_++;

  std::vector<std::string> lines;
  lines.emplace_back("============================================================");
  {
    std::ostringstream oss;
    oss << "[UWB_ANCHOR_LAYOUT] READY source=" << source
        << " frame=" << frame_name
        << " layout_version=" << anchor_layout_version_
        << " anchor_count=" << ordered_ids.size();
    lines.push_back(oss.str());
  }
  for (int id : ordered_ids)
  {
    const auto &anchor = anchors_.at(id);
    const std::string role = !anchor.role.empty() ? anchor.role :
                             (id == baseline_anchor_start_id_ ? "entry" :
                              (id == baseline_anchor_end_id_ ? "exit" : "anchor"));
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "id=" << id << " role=" << role
        << " enabled=" << static_cast<int>(anchor.enabled)
        << " estimated=" << static_cast<int>(anchor.estimated)
        << " range_bias_m=" << anchor.range_bias_m
        << " position=(" << anchor.position_w.x() << "," << anchor.position_w.y() << ","
        << anchor.position_w.z() << ")m";
    lines.push_back(oss.str());
  }

  const auto start_it = anchors_.find(baseline_anchor_start_id_);
  const auto end_it = anchors_.find(baseline_anchor_end_id_);
  if (start_it != anchors_.end() && end_it != anchors_.end())
  {
    const V3D baseline = end_it->second.position_w - start_it->second.position_w;
    const double length = baseline.norm();
    V3D direction = V3D::Zero();
    if (length > 1e-12) direction = baseline / length;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "baseline_start_id=" << baseline_anchor_start_id_
        << " baseline_end_id=" << baseline_anchor_end_id_
        << " baseline_length=" << length << "m"
        << " baseline_direction=(" << direction.x() << "," << direction.y() << ","
        << direction.z() << ")";
    lines.push_back(oss.str());
  }
  lines.emplace_back("============================================================");

  if (report != nullptr)
  {
    report->deferred_result_lines.insert(report->deferred_result_lines.end(), lines.begin(), lines.end());
  }
  else
  {
    const double stamp = ros::Time::now().toSec();
    for (const auto &line : lines) emitUwbLine(stamp, "INFO", line, true, true);
  }
}

void UwbManager::logFinalTagOffset(const std::string &source, UwbUpdateReport *report)
{
  const V3D estimated_add = tag_offset_est_body_ - tag_offset_body_;
  std::vector<std::string> lines;
  lines.emplace_back("============================================================");
  {
    std::ostringstream oss;
    oss << "[UWB_TAG_OFFSET] READY source=" << source
        << " estimate_version=" << tag_offset_estimate_version_
        << " frame=body direction=body_origin_to_uwb_tag";
    lines.push_back(oss.str());
  }
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "configured_translation=(" << tag_offset_body_.x() << "," << tag_offset_body_.y() << ","
        << tag_offset_body_.z() << ")m"
        << " estimated_translation_add=(" << estimated_add.x() << "," << estimated_add.y() << ","
        << estimated_add.z() << ")m"
        << " final_translation=(" << tag_offset_est_body_.x() << "," << tag_offset_est_body_.y() << ","
        << tag_offset_est_body_.z() << ")m"
        << " translation_norm=" << tag_offset_est_body_.norm() << "m";
    lines.push_back(oss.str());
  }
  lines.emplace_back("============================================================");
  if (report != nullptr)
  {
    report->deferred_result_lines.insert(report->deferred_result_lines.end(), lines.begin(), lines.end());
  }
  else
  {
    const double stamp = ros::Time::now().toSec();
    for (const auto &line : lines) emitUwbLine(stamp, "INFO", line, true, true);
  }
}

void UwbManager::finalizeUwbUpdateAttempt(UwbUpdateReport &report, const StatesGroup &state,
                                          UwbUpdateResult &result)
{
  std::set<int> seen_rejected_anchor_ids;
  report.rejected_anchor_ids.erase(
      std::remove_if(report.rejected_anchor_ids.begin(), report.rejected_anchor_ids.end(),
                     [&](int id) { return !seen_rejected_anchor_ids.insert(id).second; }),
      report.rejected_anchor_ids.end());
  report.system_position_after = state.pos_end;
  report.applied_position_correction = report.system_position_after - report.system_position_before;
  report.correction_norm = report.applied_position_correction.norm();
  report.state_updated = report.correction_norm > update_epsilon_;
  report.covariance_updated =
      (state.cov - report.covariance_before).cwiseAbs().maxCoeff() > 1e-14;
  if (!update_en_)
  {
    const V3D position_cov_before =
        report.covariance_before.block<3, 3>(3, 3).diagonal();
    const V3D position_cov_after = state.cov.block<3, 3>(3, 3).diagonal();
    std::ostringstream oss;
    oss << "[UWB_DIAGNOSTIC_GUARD] attempt=" << report.attempt_id
        << " update_en=0"
        << " state_updated=" << static_cast<int>(report.state_updated)
        << " covariance_updated=" << static_cast<int>(report.covariance_updated)
        << " position_cov_before=(" << position_cov_before.transpose() << ")"
        << " position_cov_after=(" << position_cov_after.transpose() << ")";
    report.deferred_debug_lines.push_back(oss.str());
  }

  if (report.state_updated)
  {
    report.status = UwbUpdateStatus::UPDATED;
    report.outcome = UwbUpdateOutcome::ACCEPTED;
    report.primary_reason = UwbRejectReason::NONE;
  }
  else if (report.status != UwbUpdateStatus::WAITING_INITIALIZATION)
  {
    report.status = UwbUpdateStatus::NOT_UPDATED;
    if (report.primary_reason == UwbRejectReason::NONE && report.action.find("update") != std::string::npos)
    {
      report.primary_reason = UwbRejectReason::CORRECTION_TOO_SMALL;
      report.outcome = UwbUpdateOutcome::SKIPPED;
    }
    else if (report.outcome == UwbUpdateOutcome::ACCEPTED)
    {
      report.outcome = UwbUpdateOutcome::SKIPPED;
    }
  }

  result.attempt_id = report.attempt_id;
  result.used_count = report.valid_anchor_count;
  result.state_updated = report.state_updated;
  result.covariance_updated = report.covariance_updated;
  result.correction_clamped = report.correction_clamped;
  result.correction_norm = report.correction_norm;
  result.xy_correction_applied = std::hypot(report.applied_position_correction.x(),
                                            report.applied_position_correction.y());

  uwb_attempt_count_++;
  if (report.state_updated)
  {
    uwb_update_count_++;
    uwb_correction_sum_m_ += report.correction_norm;
    uwb_correction_max_m_ = std::max(uwb_correction_max_m_, report.correction_norm);
  }
  else if (report.outcome == UwbUpdateOutcome::REJECTED)
  {
    uwb_reject_count_++;
  }
  else
  {
    uwb_skip_count_++;
  }

  if (summary_log_en_)
  {
    emitUwbLine(report.slam_stamp, "INFO", formatUwbResultLine(report),
                summary_to_console_, summary_to_file_);
  }

  if (range_debug_log_en_)
  {
    for (const auto &range : report.range_debug)
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6)
          << "[UWB_RANGE] attempt=" << report.attempt_id
          << " anchor=" << range.anchor_id
          << " measurement_uid=" << range.measurement_uid
          << " source_line=" << range.source_line
          << " diag=" << range.diag
          << " source_format=" << range.source_format
          << " consumed_after_attempt=1"
          << " raw_range=" << range.raw_range_m << "m"
          << " range_bias=" << range.range_bias_m << "m"
          << " corrected_range=" << range.corrected_range_m << "m"
          << " predicted_range=" << range.predicted_range_m << "m"
          << " residual=" << range.residual_m << "m"
          << " anchor_position=(" << range.anchor_position_w.x() << "," << range.anchor_position_w.y()
          << "," << range.anchor_position_w.z() << ")m"
          << " tag_position=(" << range.tag_position_w.x() << "," << range.tag_position_w.y()
          << "," << range.tag_position_w.z() << ")m"
          << " time_diff=" << range.time_diff_s << "s"
          << " accepted=" << static_cast<int>(range.accepted)
          << " reject_reason=" << uwbRejectReasonName(range.reject_reason);
      emitUwbLine(report.slam_stamp, range.accepted ? "INFO" : "WARN", oss.str(),
                  debug_log_en_ && debug_to_console_, debug_log_en_ && debug_to_file_);
    }
  }
  if (debug_log_en_)
  {
    for (const auto &line : report.deferred_debug_lines)
      emitUwbLine(report.slam_stamp, "INFO", line, debug_to_console_, debug_to_file_);
  }
  for (const auto &line : report.deferred_result_lines)
    emitUwbLine(report.slam_stamp, "INFO", line, true, true);

  if (statistics_log_interval_ > 0 && uwb_attempt_count_ % statistics_log_interval_ == 0)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "[UWB_STATISTICS] attempts=" << uwb_attempt_count_
        << " updates=" << uwb_update_count_
        << " rejects=" << uwb_reject_count_
        << " skips=" << uwb_skip_count_
        << " update_ratio=" << (uwb_attempt_count_ > 0 ?
            static_cast<double>(uwb_update_count_) / static_cast<double>(uwb_attempt_count_) : 0.0)
        << " mean_correction=" << (uwb_update_count_ > 0 ?
            uwb_correction_sum_m_ / static_cast<double>(uwb_update_count_) : 0.0) << "m"
        << " max_correction=" << uwb_correction_max_m_ << "m"
        << " replay_consumed=" << replay_consumed_measurement_count_
        << " replay_stale_dropped=" << replay_stale_measurement_count_;
    emitUwbLine(report.slam_stamp, "INFO", oss.str(), summary_to_console_, summary_to_file_);
  }
}

void UwbManager::logAnchorEstimate(int anchor_id, const V3D &position_w, double rmse, int rank, int sample_count)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!update_log_file_.is_open()) return;

  update_log_file_ << std::fixed << std::setprecision(6)
                   << ros::Time::now().toSec()
                   << " ANCHOR_ESTIMATE id=" << anchor_id
                   << " position=" << position_w.transpose()
                   << " rmse=" << rmse
                   << " rank=" << rank
                   << " samples=" << sample_count
                   << '\n';
  update_log_file_.flush();
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
                                     const std::vector<UwbRangeMeasurement> &measurements,
                                     UwbUpdateReport *report)
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
      if (report != nullptr)
      {
        report->current_motion_m = trajectory_motion;
        report->required_motion_m = anchor_frame_align_min_motion_m_;
        report->deferred_debug_lines.push_back(
            "[UWB_ANCHOR_ESTIMATION] attempt=" + std::to_string(report->attempt_id) +
            " status=WAITING " + oss.str());
      }
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
    if (report != nullptr)
      report->deferred_debug_lines.push_back(
          "[UWB_ANCHOR_ESTIMATION] attempt=" + std::to_string(report->attempt_id) +
          " status=ESTIMATING " + oss.str());

    for (const auto &item : pending_aligned_anchors_)
    {
      std::ostringstream anchor_oss;
      anchor_oss << "ANCHOR_FRAME_ALIGN_CANDIDATE_ANCHOR id=" << item.first
                 << " manual=" << configured_anchors_[item.first].position_w.transpose()
                 << " aligned=" << item.second.position_w.transpose();
      if (report != nullptr) report->deferred_debug_lines.push_back(anchor_oss.str());
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
    if (report != nullptr)
      report->deferred_debug_lines.push_back(
          "[UWB_ANCHOR_ESTIMATION] attempt=" + std::to_string(report->attempt_id) +
          " status=ESTIMATING " + oss.str());
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
    if (report != nullptr)
      report->deferred_result_lines.push_back(
          "[UWB_ANCHOR_ESTIMATION] status=FAILED " + oss.str());
    for (const auto &item : pending_aligned_anchors_)
    {
      std::ostringstream anchor_oss;
      anchor_oss << "ANCHOR_FRAME_ALIGN_FAILED_ANCHOR id=" << item.first
                 << " manual=" << configured_anchors_[item.first].position_w.transpose()
                 << " aligned=" << item.second.position_w.transpose();
      if (report != nullptr) report->deferred_debug_lines.push_back(anchor_oss.str());
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
  if (report != nullptr)
    report->deferred_result_lines.push_back(
        "[UWB_ANCHOR_ESTIMATION] status=READY source=anchor_frame_alignment");
  for (const auto &item : anchors_)
  {
    std::ostringstream anchor_oss;
    anchor_oss << "ANCHOR_FRAME_ALIGNED_ANCHOR id=" << item.first
               << " position=" << item.second.position_w.transpose();
    if (report != nullptr) report->deferred_debug_lines.push_back(anchor_oss.str());
  }
  logFinalAnchorLayout("anchor_frame_alignment", "camera_init", report);
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
                                              const std::vector<UwbRangeMeasurement> &measurements,
                                              UwbUpdateReport *report)
{
  if (!baseline_anchor_init_en_) return false;
  if (baseline_anchors_initialized_) return true;

  const double baseline_distance = configuredBaselineDistance();
  if (baseline_distance <= 0.0)
  {
    if (report != nullptr)
      report->deferred_debug_lines.push_back(
          "[UWB_ANCHOR_ESTIMATION] attempt=" + std::to_string(report->attempt_id) +
          " status=WAITING reason=missing_baseline_distance");
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
  if (report != nullptr)
  {
    report->current_motion_m = motion_norm;
    report->required_motion_m = baseline_init_min_motion_m_;
  }
  if (motion_norm < baseline_init_min_motion_m_)
  {
    std::ostringstream oss;
    oss << "WAIT_BASELINE_ANCHORS motion=" << motion_norm
        << " min_motion=" << baseline_init_min_motion_m_
        << " distance=" << baseline_distance;
    if (report != nullptr)
      report->deferred_debug_lines.push_back(
          "[UWB_ANCHOR_ESTIMATION] attempt=" + std::to_string(report->attempt_id) +
          " status=WAITING " + oss.str());
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
  const auto configured_start_it = configured_anchors_.find(baseline_anchor_start_id_);
  if (configured_start_it != configured_anchors_.end()) start_anchor = configured_start_it->second;
  start_anchor.id = baseline_anchor_start_id_;
  start_anchor.enabled = true;
  start_anchor.estimated = true;
  start_anchor.position_w = start_anchor_pos;

  UwbAnchor end_anchor;
  const auto configured_end_it = configured_anchors_.find(baseline_anchor_end_id_);
  if (configured_end_it != configured_anchors_.end()) end_anchor = configured_end_it->second;
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
  if (report != nullptr)
    report->deferred_result_lines.push_back(
        "[UWB_ANCHOR_ESTIMATION] status=READY source=two_anchor_baseline_initialization");
  logFinalAnchorLayout("two_anchor_baseline_initialization", "camera_init", report);
  return true;
}

void UwbManager::collectAnchorEstimateSamples(const StatesGroup &state,
                                              const std::vector<UwbRangeMeasurement> &measurements,
                                              UwbUpdateReport *report)
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
        if (report != nullptr)
          report->deferred_debug_lines.push_back(
              "[UWB_ANCHOR_ESTIMATION] attempt=" + std::to_string(report->attempt_id) +
              " status=ESTIMATING " + oss.str());
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
    if (report != nullptr)
    {
      std::ostringstream oss;
      oss << "[UWB_ANCHOR_ESTIMATION] status=READY source=anchor_position_estimation"
          << " id=" << anchor_id
          << " position=(" << logged_position.transpose() << ")m"
          << " rmse=" << rmse << "m rank=" << rank
          << " samples=" << item.second.size();
      report->deferred_result_lines.push_back(oss.str());
    }
    else
    {
      logAnchorEstimate(anchor_id, logged_position, rmse, rank,
                        static_cast<int>(item.second.size()));
    }
    if (anchor_estimate_use_for_update_)
      logFinalAnchorLayout("anchor_position_estimation", "camera_init", report);
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

double UwbManager::effectivePositionCovFloor() const
{
  const bool is_degraded = degraded_mode_en_ && degraded_mode_;
  return selectUwbPositionCovFloor(position_cov_floor_m_, position_cov_floor_degraded_m_,
                                   position_cov_floor_degraded_only_, is_degraded);
}

bool UwbManager::solveUwbOnlyPosition2D(const std::vector<UwbRangeMeasurement> &measurements,
                                        double z_world, const V3D &initial_position,
                                        V3D &position, double &residual_rms,
                                        double &max_abs_residual, double &geometry_score) const
{
  if (measurements.size() < 3) return false;

  V2D xy(initial_position.x(), initial_position.y());
  int valid_count = 0;
  Eigen::Matrix2d last_A = Eigen::Matrix2d::Zero();
  for (int iter = 0; iter < 12; ++iter)
  {
    Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b = Eigen::Vector2d::Zero();
    valid_count = 0;
    for (const auto &measurement : measurements)
    {
      const auto anchor_it = anchors_.find(measurement.anchor_id);
      if (anchor_it == anchors_.end()) continue;
      const V3D &anchor = anchor_it->second.position_w;
      const double dx = xy.x() - anchor.x();
      const double dy = xy.y() - anchor.y();
      const double dz = z_world - anchor.z();
      const double predicted = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (predicted < 1e-6 || !std::isfinite(predicted)) continue;
      const double r = predicted - measurement.range_m;
      const double abs_r = std::fabs(r);
      const double huber = 1.0;
      const double w = abs_r <= huber ? 1.0 : huber / std::max(abs_r, 1e-9);
      Eigen::Vector2d J(dx / predicted, dy / predicted);
      A += w * (J * J.transpose());
      b += -w * J * r;
      valid_count++;
    }
    if (valid_count < 3) return false;
    last_A = A;
    Eigen::LDLT<Eigen::Matrix2d> ldlt(A);
    if (ldlt.info() != Eigen::Success) return false;
    const Eigen::Vector2d delta = ldlt.solve(b);
    if (!delta.allFinite()) return false;
    xy += delta;
    if (delta.norm() < 1e-4) break;
  }

  position = initial_position;
  position.x() = xy.x();
  position.y() = xy.y();
  residual_rms = 0.0;
  max_abs_residual = 0.0;
  valid_count = 0;
  for (const auto &measurement : measurements)
  {
    const auto anchor_it = anchors_.find(measurement.anchor_id);
    if (anchor_it == anchors_.end()) continue;
    const double predicted = (position - anchor_it->second.position_w).norm();
    if (predicted < 1e-6 || !std::isfinite(predicted)) continue;
    const double r = predicted - measurement.range_m;
    residual_rms += r * r;
    max_abs_residual = std::max(max_abs_residual, std::fabs(r));
    valid_count++;
  }
  if (valid_count < 3) return false;
  residual_rms = std::sqrt(residual_rms / static_cast<double>(valid_count));

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(last_A);
  if (eig.info() != Eigen::Success) return false;
  const double l_min = std::max(0.0, eig.eigenvalues()(0));
  const double l_max = std::max(l_min, eig.eigenvalues()(1));
  geometry_score = l_max > 1e-9 ? l_min / l_max : 0.0;
  return geometry_score >= anchor_geometry_min_score_;
}

V3D UwbManager::updateFilteredUwbOnlyPosition(const V3D &position, double stamp,
                                              double &position_jump, double &speed)
{
  position_jump = 0.0;
  speed = 0.0;
  if (!uwb_only_position_history_.empty())
  {
    const auto &last = uwb_only_position_history_.back();
    position_jump = std::hypot(position.x() - last.position.x(), position.y() - last.position.y());
    const double dt = stamp - last.stamp;
    if (dt > 1e-6) speed = position_jump / dt;
  }

  uwb_only_position_history_.push_back({position, stamp});
  while (uwb_only_position_history_.size() > 5) uwb_only_position_history_.pop_front();

  std::vector<double> xs;
  std::vector<double> ys;
  xs.reserve(uwb_only_position_history_.size());
  ys.reserve(uwb_only_position_history_.size());
  for (const auto &sample : uwb_only_position_history_)
  {
    xs.push_back(sample.position.x());
    ys.push_back(sample.position.y());
  }
  std::sort(xs.begin(), xs.end());
  std::sort(ys.begin(), ys.end());
  const size_t mid = xs.size() / 2;
  V3D filtered = position;
  filtered.x() = xs[mid];
  filtered.y() = ys[mid];
  return filtered;
}

UwbUpdateResult UwbManager::applyRangeUpdate(StatesGroup &state)
{
  UwbUpdateResult result;
  if (!en_) return result;
  const std::string source = toLower(input_source_);
  if (source == "file" || source == "txt" || source == "replay")
  {
    ROS_WARN_THROTTLE(3.0,
                      "[UWB] File replay requires SLAM/LiDAR timestamps; skip wall-time applyRangeUpdate().");
    logEventThrottled(ros::Time::now().toSec(), "skip_wall_time_replay", 3.0, "WARN",
                      "SKIP_WALL_TIME_REPLAY reason=file_replay_requires_applyRangeUpdateAt");
    result.action = "skip_wall_time_replay";
    return result;
  }

  const double now = ros::Time::now().toSec();
  return applyRangeUpdateAt(state, now, now);
}

UwbUpdateResult UwbManager::applyRangeUpdateAt(StatesGroup &state, double current_lidar_stamp, double lidar_start_stamp)
{
  UwbUpdateResult result;
  if (!en_) return result;
  const std::string source = toLower(input_source_);
  const bool replay_source = source == "file" || source == "txt" || source == "replay";
  const double now = replay_source ? current_lidar_stamp : ros::Time::now().toSec();
  const auto received_measurements =
      replay_source ? takeReplayMeasurements(current_lidar_stamp, lidar_start_stamp) :
                      takeRecentMeasurements(now);
  if (received_measurements.empty())
  {
    result.action = "no_measurements";
    return result;
  }

  UwbUpdateReport report;
  report.attempt_id = ++uwb_update_attempt_id_;
  report.slam_stamp = current_lidar_stamp;
  report.system_position_before = state.pos_end;
  report.system_position_after = state.pos_end;
  report.covariance_before = state.cov;
  {
    std::set<int> received_ids;
    for (const auto &measurement : received_measurements) received_ids.insert(measurement.anchor_id);
    report.received_anchor_ids.assign(received_ids.begin(), received_ids.end());
  }

  const V3D tag_offset_used = tag_offset_estimate_en_ ? tag_offset_est_body_ : tag_offset_body_;
  const V3D tag_position_w = state.pos_end + state.rot_end * tag_offset_used;
  std::vector<UwbRangeMeasurement> measurements;
  measurements.reserve(received_measurements.size());
  for (const auto &measurement : received_measurements)
  {
    if (measurement.range_valid)
    {
      measurements.push_back(measurement);
      continue;
    }

    UwbRangeDebugInfo debug;
    debug.anchor_id = measurement.anchor_id;
    debug.measurement_uid = measurement.measurement_uid;
    debug.source_line = measurement.source_line;
    debug.diag = measurement.diag;
    debug.source_format = measurement.source_format;
    debug.raw_range_m = measurement.raw_range_m;
    debug.range_bias_m = measurement.range_bias_m;
    debug.corrected_range_m = measurement.range_m;
    debug.time_diff_s = measurement.time_diff_s;
    debug.tag_position_w = tag_position_w;
    const auto configured_it = configured_anchors_.find(measurement.anchor_id);
    if (configured_it != configured_anchors_.end())
      debug.anchor_position_w = configured_it->second.position_w;
    debug.reject_reason = measurement.range_reject_reason;
    report.range_debug.push_back(debug);
    report.rejected_anchor_ids.push_back(measurement.anchor_id);
    if (report.primary_reason == UwbRejectReason::NONE)
      report.primary_reason = measurement.range_reject_reason;
  }
  if (measurements.empty())
  {
    result.action = "skip_invalid_ranges";
    report.action = result.action;
    report.mode = "range_validation";
    report.outcome = UwbUpdateOutcome::REJECTED;
    report.valid_anchor_count = 0;
    finalizeUwbUpdateAttempt(report, state, result);
    return result;
  }

  if ((anchor_frame_align_en_ && !anchor_frame_align_start_pose_ready_) ||
      (!anchor_frame_align_en_ && baseline_anchor_init_en_ && !baseline_start_pose_ready_))
  {
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

  const bool anchor_frame_ready = tryAlignAnchorFrame(state, measurements, &report);
  const bool baseline_ready = anchor_frame_align_en_ ? false :
                              tryInitializeBaselineAnchors(state, measurements, &report);
  collectAnchorEstimateSamples(state, measurements, &report);
  if (anchors_.empty())
  {
    if (anchor_frame_align_en_ && !anchor_frame_ready)
    {
      report.mode = "anchor_frame_align";
      report.action = "wait_anchor_frame_alignment";
      report.status = UwbUpdateStatus::WAITING_INITIALIZATION;
      report.outcome = UwbUpdateOutcome::WAITING;
      report.primary_reason = UwbRejectReason::ANCHORS_NOT_READY;
    }
    else if (baseline_anchor_init_en_ && !baseline_ready)
    {
      report.mode = "two_anchor_baseline_initialization";
      report.action = "wait_baseline_anchors";
      report.status = UwbUpdateStatus::WAITING_INITIALIZATION;
      report.outcome = UwbUpdateOutcome::WAITING;
      report.primary_reason = UwbRejectReason::BASELINE_NOT_INITIALIZED;
    }
    else if (anchor_position_estimate_en_)
    {
      report.mode = "anchor_position_estimate";
      report.action = "wait_anchor_estimate";
      report.status = UwbUpdateStatus::WAITING_INITIALIZATION;
      report.outcome = UwbUpdateOutcome::WAITING;
      report.primary_reason = UwbRejectReason::ANCHORS_NOT_READY;
    }
    else
    {
      report.mode = "no_anchor_positions";
      report.action = "skip_no_anchor_positions";
      report.outcome = UwbUpdateOutcome::SKIPPED;
      report.primary_reason = UwbRejectReason::ANCHORS_NOT_READY;
    }
    result.used_count = static_cast<int>(measurements.size());
    result.action = "skip_no_anchor_positions";
    finalizeUwbUpdateAttempt(report, state, result);
    return result;
  }
  result = applyLatestMeasurements(state, measurements, report);
  finalizeUwbUpdateAttempt(report, state, result);
  return result;
}

UwbUpdateResult UwbManager::applyLatestMeasurements(StatesGroup &state,
                                                    const std::vector<UwbRangeMeasurement> &measurements,
                                                    UwbUpdateReport &report)
{
  UwbUpdateResult result;
  const int required_anchors = std::max(1, min_anchors_for_update_);
  report.required_anchor_count = required_anchors;
  int total_configured_anchors = 0;
  for (const auto &item : configured_anchors_)
  {
    if (item.second.enabled) total_configured_anchors++;
  }
  std::map<int, UwbRangeMeasurement> latest_by_anchor;
  for (const auto &measurement : measurements)
  {
    if (anchors_.find(measurement.anchor_id) == anchors_.end())
    {
      UwbRangeDebugInfo debug;
      debug.anchor_id = measurement.anchor_id;
      debug.measurement_uid = measurement.measurement_uid;
      debug.source_line = measurement.source_line;
      debug.diag = measurement.diag;
      debug.source_format = measurement.source_format;
      debug.raw_range_m = measurement.raw_range_m;
      debug.range_bias_m = measurement.range_bias_m;
      debug.corrected_range_m = measurement.range_m;
      debug.time_diff_s = measurement.time_diff_s;
      debug.tag_position_w = state.pos_end;
      debug.reject_reason = UwbRejectReason::UNKNOWN_ANCHOR;
      report.range_debug.push_back(debug);
      report.rejected_anchor_ids.push_back(measurement.anchor_id);
      continue;
    }
    latest_by_anchor[measurement.anchor_id] = measurement;
  }
  const bool single_anchor_entry_allowed =
      single_anchor_corridor_1d_en_ &&
      static_cast<int>(latest_by_anchor.size()) == 1 &&
      (!single_anchor_only_when_total_anchors_eq_2_ || total_configured_anchors == 2);
  const bool two_anchor_gt2_policy_entry_allowed =
      total_configured_anchors >= 3 &&
      static_cast<int>(latest_by_anchor.size()) == 2 &&
      two_anchor_policy_when_total_anchors_gt2_ != "disable";
  if (static_cast<int>(latest_by_anchor.size()) < required_anchors &&
      !single_anchor_entry_allowed &&
      !two_anchor_gt2_policy_entry_allowed)
  {
    result.used_count = static_cast<int>(latest_by_anchor.size());
    result.action = "skip_not_enough_anchors";
    report.action = result.action;
    report.mode = "anchor_count_gate";
    report.outcome = UwbUpdateOutcome::SKIPPED;
    report.primary_reason = UwbRejectReason::NOT_ENOUGH_VALID_ANCHORS;
    report.valid_anchor_count = result.used_count;
    for (const auto &item : latest_by_anchor) report.used_anchor_ids.push_back(item.first);
    return result;
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
  std::vector<UwbRangeMeasurement> used_measurements;
  for (const auto &measurement : usable_measurements)
  {
    const auto anchor_it = anchors_.find(measurement.anchor_id);
    if (anchor_it == anchors_.end()) continue;

    UwbRangeDebugInfo range_debug;
    range_debug.anchor_id = measurement.anchor_id;
    range_debug.measurement_uid = measurement.measurement_uid;
    range_debug.source_line = measurement.source_line;
    range_debug.diag = measurement.diag;
    range_debug.source_format = measurement.source_format;
    range_debug.raw_range_m = measurement.raw_range_m;
    range_debug.range_bias_m = measurement.range_bias_m;
    range_debug.corrected_range_m = measurement.range_m;
    range_debug.time_diff_s = measurement.time_diff_s;
    range_debug.anchor_position_w = anchor_it->second.position_w;
    range_debug.tag_position_w = tag_position_w;

    const V3D diff = tag_position_w - anchor_it->second.position_w;
    const double predicted_3d = diff.norm();
    const double predicted_xy = std::hypot(diff.x(), diff.y());
    const double height_diff = diff.z();
    const double predicted_range = use_3d_range_model_ ? predicted_3d : predicted_xy;
    range_debug.predicted_range_m = predicted_range;
    if (predicted_range < 1e-6)
    {
      range_debug.reject_reason = UwbRejectReason::RANGE_LIMIT;
      report.range_debug.push_back(range_debug);
      report.rejected_anchor_ids.push_back(measurement.anchor_id);
      continue;
    }

    const double residual = predicted_range - measurement.range_m;
    range_debug.residual_m = residual;
    if (!std::isfinite(residual))
    {
      range_debug.reject_reason = UwbRejectReason::INVALID_RANGE_STATUS;
      report.range_debug.push_back(range_debug);
      report.rejected_anchor_ids.push_back(measurement.anchor_id);
      continue;
    }
    if (max_residual_m_ > 0.0 && std::fabs(residual) > max_residual_m_)
    {
      range_debug.reject_reason = UwbRejectReason::RANGE_RESIDUAL_GATE;
      report.range_debug.push_back(range_debug);
      report.rejected_anchor_ids.push_back(measurement.anchor_id);
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
      range_debug.reject_reason = UwbRejectReason::LOW_GEOMETRY;
      report.range_debug.push_back(range_debug);
      report.rejected_anchor_ids.push_back(measurement.anchor_id);
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
    used_measurements.push_back(measurement);
    range_debug.accepted = true;
    range_debug.reject_reason = UwbRejectReason::NONE;
    report.range_debug.push_back(range_debug);
    row++;
  }

  const bool single_anchor_row_allowed = single_anchor_entry_allowed && row == 1;
  const bool two_anchor_gt2_policy_row_allowed =
      two_anchor_gt2_policy_entry_allowed && row == 2;
  if (row < required_anchors && !single_anchor_row_allowed && !two_anchor_gt2_policy_row_allowed)
  {
    result.used_count = row;
    result.action = "skip_not_enough_anchors";
    result.residual_rms = row > 0 ? std::sqrt(residual_sq_sum / static_cast<double>(row)) : 0.0;
    result.max_abs_residual = max_abs_residual;
    result.time_diff = time_diff_for_log;
    report.action = result.action;
    report.mode = "range_gate";
    report.outcome = UwbUpdateOutcome::SKIPPED;
    report.primary_reason = UwbRejectReason::NOT_ENOUGH_VALID_ANCHORS;
    report.used_anchor_ids = used_anchor_ids;
    report.valid_anchor_count = row;
    report.residual_rms = result.residual_rms;
    report.max_abs_residual = max_abs_residual;
    return result;
  }
  H.conservativeResize(row, DIM_STATE);
  H_tag.conservativeResize(row, 3);
  z.conservativeResize(row);
  const int used_anchor_count = static_cast<int>(used_anchor_ids.size());
  report.used_anchor_ids = used_anchor_ids;
  report.valid_anchor_count = used_anchor_count;
  const bool two_anchor_case = used_anchor_count == 2 && row == 2;
  const bool single_anchor_case = used_anchor_count == 1 && row == 1;
  const bool two_anchor_update_disabled = two_anchor_case && min_anchors_for_update_ > 2;
  const double residual_rms = std::sqrt(residual_sq_sum / static_cast<double>(row));
  double baseline_consistency_error = 0.0;
  const auto baseline_start_anchor_it = anchors_.find(baseline_anchor_start_id_);
  const auto baseline_end_anchor_it = anchors_.find(baseline_anchor_end_id_);
  const bool baseline_pair_available =
      baseline_start_anchor_it != anchors_.end() && baseline_end_anchor_it != anchors_.end();
  const bool baseline_initialized_for_update =
      baseline_pair_available &&
      (baseline_anchors_initialized_ || !baseline_anchor_init_en_ || !single_anchor_requires_baseline_initialized_);
  V3D baseline_direction = V3D::Zero();
  V3D baseline_start_position = V3D::Zero();
  V3D baseline_end_position = V3D::Zero();
  double baseline_length = 0.0;
  if (baseline_pair_available)
  {
    baseline_start_position = baseline_start_anchor_it->second.position_w;
    baseline_end_position = baseline_end_anchor_it->second.position_w;
    const V3D baseline_vec = baseline_end_position - baseline_start_position;
    baseline_length = baseline_vec.norm();
    if (baseline_length > 1e-6) baseline_direction = baseline_vec / baseline_length;
    if (update_xy_only_ || !update_z_)
    {
      const double dir_xy_norm = std::hypot(baseline_direction.x(), baseline_direction.y());
      if (dir_xy_norm > 1e-6)
      {
        baseline_direction << baseline_direction.x() / dir_xy_norm,
                              baseline_direction.y() / dir_xy_norm,
                              0.0;
      }
    }
  }
  double baseline_s_pred = 0.0;
  double baseline_s_meas = 0.0;
  double baseline_residual = 0.0;
  double baseline_residual_after_gate = 0.0;
  bool two_anchor_uses_baseline_pair = false;
  if (two_anchor_case && used_measurements.size() == 2)
  {
    const UwbRangeMeasurement *start_measurement = nullptr;
    const UwbRangeMeasurement *end_measurement = nullptr;
    for (const auto &measurement : used_measurements)
    {
      if (measurement.anchor_id == baseline_anchor_start_id_) start_measurement = &measurement;
      if (measurement.anchor_id == baseline_anchor_end_id_) end_measurement = &measurement;
    }
    two_anchor_uses_baseline_pair = start_measurement != nullptr && end_measurement != nullptr;
    if (baseline_pair_available && baseline_length > 1e-6 && two_anchor_uses_baseline_pair)
    {
      baseline_consistency_error = std::fabs((start_measurement->range_m + end_measurement->range_m) -
                                             baseline_length);

      const bool baseline_1d_mode =
          (total_configured_anchors == 2 ||
           two_anchor_policy_when_total_anchors_gt2_ == "baseline_1d_only_if_pair_matches_corridor") &&
          (two_anchor_update_mode_ == "baseline_1d" ||
           two_anchor_update_mode_ == "baseline_1d_direct");
      if (baseline_1d_mode)
      {
        const double d_start = start_measurement->range_m;
        const double d_end = end_measurement->range_m;
        baseline_s_meas = (d_start * d_start + baseline_length * baseline_length - d_end * d_end) /
                          (2.0 * baseline_length);
        baseline_s_pred = (tag_position_w - baseline_start_position).dot(baseline_direction);
        baseline_residual = baseline_s_meas - baseline_s_pred;
        baseline_residual_after_gate = baseline_residual;
        H = Eigen::MatrixXd::Zero(1, DIM_STATE);
        H_tag = Eigen::MatrixXd::Zero(1, 3);
        z = Eigen::VectorXd::Zero(1);
        H(0, 3) = baseline_direction.x();
        H(0, 4) = baseline_direction.y();
        H(0, 5) = 0.0;
        z(0) = baseline_residual;
        row = 1;
      }
    }
  }
  const double effective_range_noise_m = range_noise_m_ * (two_anchor_case ? two_anchor_sigma_scale_ : 1.0);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(row, row) * (effective_range_noise_m * effective_range_noise_m);
  const bool h_orientation_zero = H.block(0, 0, row, 3).cwiseAbs().maxCoeff() < 1e-12;
  const bool h_z_zero = H.col(5).cwiseAbs().maxCoeff() < 1e-12;

  const bool estimate_tag_offset_this_update =
      !update_xy_only_ && tag_offset_estimate_en_ && row >= tag_offset_estimate_min_anchors_;

  if (!estimate_tag_offset_this_update)
  {
    if (tag_offset_estimate_en_ && update_xy_only_)
    {
      report.deferred_debug_lines.push_back(
          "[UWB_TAG_OFFSET_ESTIMATION] attempt=" + std::to_string(report.attempt_id) +
          " status=WAITING reason=xy_only_update");
    }
    else if (tag_offset_estimate_en_)
    {
      std::ostringstream oss;
      oss << "[UWB_TAG_OFFSET_ESTIMATION] attempt=" << report.attempt_id
          << " status=WAITING used_anchors=" << row
          << " required=" << tag_offset_estimate_min_anchors_
          << " pose_update_still_runs=1";
      report.deferred_debug_lines.push_back(oss.str());
    }

    MD(DIM_STATE, DIM_STATE) cov_for_uwb = state.cov;
    const double position_cov_floor_used = effectivePositionCovFloor();
    const bool covariance_degraded = degraded_mode_en_ && degraded_mode_;
    bool covariance_inflated = false;
    if (position_cov_floor_used > 0.0)
    {
      const double floor_var = position_cov_floor_used * position_cov_floor_used;
      const int pos_dims = update_z_ ? 3 : 2;
      for (int i = 0; i < pos_dims; ++i)
      {
        const int idx = 3 + i;
        if (cov_for_uwb(idx, idx) < floor_var)
        {
          cov_for_uwb(idx, idx) = floor_var;
          covariance_inflated = true;
        }
      }
    }

    const Eigen::MatrixXd S = H * cov_for_uwb * H.transpose() + R;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success)
    {
      result.action = "skip_cov_decomposition";
      report.action = result.action;
      report.mode = two_anchor_case ? two_anchor_update_mode_ :
                    (single_anchor_case ? "single_anchor_corridor_1d" : "multi_anchor");
      report.outcome = UwbUpdateOutcome::REJECTED;
      report.primary_reason = UwbRejectReason::COVARIANCE_INVALID;
      return result;
    }

    const Eigen::MatrixXd K = cov_for_uwb * H.transpose() * ldlt.solve(Eigen::MatrixXd::Identity(row, row));
    Eigen::VectorXd dx_unlimited = K * z;
    if (dx_unlimited.size() != DIM_STATE || !dx_unlimited.allFinite())
    {
      result.action = "skip_non_finite_dx";
      report.action = result.action;
      report.outcome = UwbUpdateOutcome::REJECTED;
      report.primary_reason = UwbRejectReason::NON_FINITE_CORRECTION;
      return result;
    }
    const double z_correction_before_clamp = dx_unlimited(5, 0);

    const bool baseline_direction_valid = std::hypot(baseline_direction.x(), baseline_direction.y()) > 1e-6;
    const bool baseline_projection_mode =
        two_anchor_case &&
        row == 1 &&
        baseline_direction_valid &&
        (two_anchor_update_mode_ == "baseline_1d" ||
         two_anchor_update_mode_ == "baseline_1d_direct");
    const bool allow_z = update_z_ && !update_xy_only_;
    const bool allow_orientation = update_orientation_ && !update_xy_only_;
    const V3D *projection_direction = baseline_projection_mode ? &baseline_direction : nullptr;
    const Eigen::MatrixXd K_constrained =
        applyUwbUpdateMaskAndProjection(K, allow_z, allow_orientation, projection_direction);
    double kalman_gain_norm = K_constrained.norm();
    Eigen::VectorXd dx_raw_dynamic = K_constrained * z;
    if (dx_raw_dynamic.size() != DIM_STATE || !dx_raw_dynamic.allFinite())
    {
      result.action = "skip_non_finite_dx";
      report.action = result.action;
      report.outcome = UwbUpdateOutcome::REJECTED;
      report.primary_reason = UwbRejectReason::NON_FINITE_CORRECTION;
      return result;
    }
    VD(DIM_STATE) dx_raw = VD(DIM_STATE)::Zero();
    dx_raw = dx_raw_dynamic;
    VD(DIM_STATE) dx_after_baseline_projection = dx_raw;
    if (baseline_projection_mode)
    {
      dx_after_baseline_projection = dx_raw;
    }
    else
    {
      dx_after_baseline_projection = dx_raw;
    }

    V3D trans_raw = dx_raw.block<3, 1>(3, 0);
    double xy_correction_raw = std::hypot(trans_raw.x(), trans_raw.y());
    result.used_count = used_anchor_count;
    result.residual_rms = residual_rms;
    result.max_abs_residual = max_abs_residual;
    result.xy_correction_raw = xy_correction_raw;
    result.time_diff = time_diff_for_log;
    result.baseline_consistency_error = baseline_consistency_error;
    result.limited_update_consecutive_good_count = limited_update_consecutive_good_count_;
    result.relocalization_candidate_count = relocalization_candidate_count_;
    bool uwb_only_valid = false;
    double uwb_only_geometry_score = 0.0;
    if (used_anchor_count >= 3)
    {
      double uwb_only_stamp = 0.0;
      for (const auto &measurement : used_measurements)
      {
        uwb_only_stamp = std::max(uwb_only_stamp, measurement.stamp);
      }
      uwb_only_valid = solveUwbOnlyPosition2D(used_measurements, tag_position_w.z(), tag_position_w,
                                              result.uwb_only_position,
                                              result.uwb_only_residual_rms,
                                              result.uwb_only_max_abs_residual,
                                              uwb_only_geometry_score);
      if (uwb_only_valid)
      {
        result.filtered_uwb_position =
            updateFilteredUwbOnlyPosition(result.uwb_only_position, uwb_only_stamp,
                                          result.uwb_only_position_jump, result.uwb_only_speed);
        result.slam_uwb_position_diff =
            std::hypot(result.filtered_uwb_position.x() - tag_position_w.x(),
                       result.filtered_uwb_position.y() - tag_position_w.y());
      }
    }
    std::ostringstream used_ids;
    for (size_t i = 0; i < used_anchor_ids.size(); ++i)
    {
      if (i > 0) used_ids << ",";
      used_ids << used_anchor_ids[i];
    }
    V3D position_cov_before_update = state.cov.block<3, 3>(3, 3).diagonal();
    V3D position_cov_after_update = position_cov_before_update;
    VD(DIM_STATE) dx_after_clamp = VD(DIM_STATE)::Zero();
    std::string update_mode_for_log = two_anchor_case ? two_anchor_update_mode_ :
                                      (single_anchor_case ? "single_anchor_corridor_1d" : "multi_anchor");
    std::string selected_update_policy = update_mode_for_log;
    bool whether_single_anchor_allowed = single_anchor_case && single_anchor_entry_allowed;
    bool whether_pair_matches_corridor = two_anchor_case && two_anchor_uses_baseline_pair;
    int single_anchor_id = single_anchor_case ? used_anchor_ids.front() : -1;
    double single_anchor_s_anchor = 0.0;
    double single_anchor_abs_s_pred_minus_s_anchor = 0.0;
    double single_anchor_measured_range = 0.0;
    double single_anchor_height_diff = 0.0;
    double single_anchor_rho = 0.0;
    double single_anchor_candidate_1 = 0.0;
    double single_anchor_candidate_2 = 0.0;
    int single_anchor_selected_branch = 0;
    int single_anchor_previous_selected_branch = single_anchor_last_branch_;
    double single_anchor_branch_margin = 0.0;
    double single_anchor_residual = 0.0;
    int single_anchor_confirm_counter_log = single_anchor_confirm_counter_;
    double single_anchor_range_jump = 0.0;
    double single_anchor_residual_jump = 0.0;
    double single_anchor_estimated_range_speed = 0.0;
    double corridor_direction_angle_deg = 0.0;
    std::string skip_reason = "none";
    std::string reason_not_confirmed = "none";
    double delta_s = 0.0;
    double selected_max_step = baseline_1d_direct_max_step_m_;
    std::string degradation_level = covariance_degraded ? "degraded" : "normal";
    bool is_degraded = covariance_degraded;
    bool is_strong_degraded = false;
    auto corridorStableCountForResidual = [&](double residual) {
      if (!corridor_last_residual_valid_ || std::fabs(corridor_last_residual_) < 1e-9 ||
          std::fabs(residual) < 1e-9 ||
          (corridor_last_residual_ > 0.0) == (residual > 0.0))
      {
        return corridor_residual_stable_count_ + 1;
      }
      return 1;
    };
    auto selectCorridorMaxStep = [&](bool single_anchor_mode, int stable_count) {
      const double normal = single_anchor_mode ? single_anchor_normal_max_step_m_ : two_anchor_normal_max_step_m_;
      const double degraded = single_anchor_mode ? single_anchor_degraded_max_step_m_ : two_anchor_degraded_max_step_m_;
      const double strong = single_anchor_mode ? single_anchor_strong_degraded_max_step_m_ :
                                                 two_anchor_strong_degraded_max_step_m_;
      const double hard = single_anchor_mode ? single_anchor_hard_max_step_m_ : two_anchor_hard_max_step_m_;
      degradation_level = covariance_degraded ? "degraded" : "normal";
      is_degraded = degraded_mode_en_ && degraded_mode_;
      is_strong_degraded = false;
      double step = normal;
      if (is_degraded && stable_count >= degraded_confirm_count_)
      {
        step = degraded;
        degradation_level = "degraded";
        if (stable_count >= strong_degraded_confirm_count_)
        {
          step = strong;
          is_strong_degraded = true;
          degradation_level = "strong_degraded";
        }
      }
      return std::min(step, hard);
    };
    auto logCovarianceStages = [&](const std::string &update_mode,
                                   const Eigen::MatrixXd &covariance_after_kalman,
                                   double max_asymmetry, double min_diagonal) {
      const V3D position_cov_after_floor =
          cov_for_uwb.block<3, 3>(3, 3).diagonal();
      const V3D position_cov_after_kalman =
          covariance_after_kalman.block<3, 3>(3, 3).diagonal();
      const V3D position_cov_final =
          state.cov.block<3, 3>(3, 3).diagonal();
      std::ostringstream oss;
      oss << "[UWB_DEBUG_COVARIANCE] attempt=" << report.attempt_id
          << " update_mode=" << update_mode
          << " degradation_level=" << degradation_level
          << " effective_sigma=" << effective_range_noise_m
          << " measurement_variance="
          << effective_range_noise_m * effective_range_noise_m
          << " kalman_gain_norm=" << kalman_gain_norm
          << " position_cov_before=(" << position_cov_before_update.transpose() << ")"
          << " position_cov_after_kalman=(" << position_cov_after_kalman.transpose() << ")"
          << " position_cov_after_floor=(" << position_cov_after_floor.transpose() << ")"
          << " position_cov_after_direct_reset=(" << position_cov_after_kalman.transpose() << ")"
          << " position_cov_final=(" << position_cov_final.transpose() << ")"
          << " position_cov_floor_used=" << position_cov_floor_used
          << " covariance_inflated=" << static_cast<int>(covariance_inflated)
          << " covariance_reset=0"
          << " covariance_reset_reason=NONE"
          << " max_asymmetry=" << max_asymmetry
          << " min_diagonal=" << min_diagonal
          << " finite=" << static_cast<int>(state.cov.allFinite());
      report.deferred_debug_lines.push_back(oss.str());
    };
    auto log_action = [&](const std::string &level, const std::string &action,
                          double xy_correction_applied, double clamp_ratio, const V3D &delta) {
      (void)level;
      report.action = action;
      report.mode = update_mode_for_log;
      report.residual_rms = residual_rms;
      report.max_abs_residual = max_abs_residual;
      report.raw_position_correction = dx_raw.block<3, 1>(3, 0);
      report.correction_clamped = clamp_ratio < 1.0 - 1e-12;
      report.primary_reason = rejectReasonFromLegacy(skip_reason, action);
      if (single_anchor_id >= 0 &&
          (report.primary_reason == UwbRejectReason::RANGE_JUMP ||
           report.primary_reason == UwbRejectReason::RESIDUAL_JUMP ||
           report.primary_reason == UwbRejectReason::RANGE_LIMIT))
      {
        for (auto &range : report.range_debug)
        {
          if (range.anchor_id != single_anchor_id) continue;
          range.accepted = false;
          range.reject_reason = report.primary_reason;
        }
        if (std::find(report.rejected_anchor_ids.begin(), report.rejected_anchor_ids.end(),
                      single_anchor_id) == report.rejected_anchor_ids.end())
          report.rejected_anchor_ids.push_back(single_anchor_id);
      }
      if (report.primary_reason == UwbRejectReason::NONE && action.find("update") != std::string::npos)
        report.outcome = UwbUpdateOutcome::ACCEPTED;
      else if (report.primary_reason == UwbRejectReason::DEBUG_ONLY ||
               report.primary_reason == UwbRejectReason::UPDATE_DISABLED ||
               report.primary_reason == UwbRejectReason::NOT_ENOUGH_VALID_ANCHORS ||
               actionIsWaiting(action))
        report.outcome = UwbUpdateOutcome::SKIPPED;
      else
        report.outcome = UwbUpdateOutcome::REJECTED;

      std::ostringstream gate;
      gate << "[UWB_DEBUG_GATE] attempt=" << report.attempt_id
           << " residual_rms=" << residual_rms
           << " max_abs_residual=" << max_abs_residual
           << " baseline_residual=" << baseline_residual
           << " baseline_consistency_error=" << baseline_consistency_error
           << " range_jump=" << single_anchor_range_jump
           << " residual_jump=" << single_anchor_residual_jump
           << " time_diff=" << time_diff_for_log
           << " primary_reason=" << uwbRejectReasonName(report.primary_reason);
      report.deferred_debug_lines.push_back(gate.str());

      std::ostringstream update;
      update << "[UWB_DEBUG_UPDATE] attempt=" << report.attempt_id
             << " action=" << action
             << " policy=" << selected_update_policy
             << " raw_delta_s=" << (baseline_residual_after_gate == 0.0 ? 0.0 : delta_s / std::max(clamp_ratio, 1e-12))
             << " applied_delta_s=" << delta_s
             << " selected_max_step=" << selected_max_step
             << " effective_sigma=" << effective_range_noise_m
             << " kalman_gain_norm=" << kalman_gain_norm
             << " clamp_ratio=" << clamp_ratio
             << " degradation_level=" << degradation_level;
      report.deferred_debug_lines.push_back(update.str());

      std::ostringstream state_line;
      state_line << "[UWB_DEBUG_STATE] attempt=" << report.attempt_id
                 << " dx_raw=(" << dx_raw.transpose() << ")"
                 << " dx_after_projection=(" << dx_after_baseline_projection.transpose() << ")"
                 << " dx_after_clamp=(" << dx_after_clamp.transpose() << ")"
                 << " position_cov_before=(" << position_cov_before_update.transpose() << ")"
                 << " position_cov_floor_normal=" << position_cov_floor_m_
                 << " position_cov_floor_degraded=" << position_cov_floor_degraded_m_
                 << " position_cov_floor_used=" << position_cov_floor_used
                 << " position_cov_after=(" << position_cov_after_update.transpose() << ")"
                 << " covariance_inflated=" << static_cast<int>(covariance_inflated)
                 << " degradation_level=" << degradation_level
                 << " H_orientation_zero=" << static_cast<int>(h_orientation_zero)
                 << " H_z_zero=" << static_cast<int>(h_z_zero)
                 << " z_correction_before_mask=" << z_correction_before_clamp
                 << " final_trans_add=(" << delta.transpose() << ")";
      report.deferred_debug_lines.push_back(state_line.str());
    };
    auto finalize_result = [&](const std::string &action, bool state_updated,
                               double xy_correction_applied, const V3D &delta,
                               bool request_relocalization = false) {
      result.action = action;
      result.xy_correction_applied = xy_correction_applied;
      result.correction_norm = delta.norm();
      result.state_updated = state_updated;
      result.covariance_inflated = state_updated && covariance_inflated;
      result.request_pause_map_insert = state_updated && xy_correction_applied > 0.05;
      result.request_relocalization = request_relocalization;
      result.limited_update_consecutive_good_count = limited_update_consecutive_good_count_;
      result.relocalization_candidate_count = relocalization_candidate_count_;
      return result;
    };
    auto rememberSingleAnchorSample = [&](const UwbRangeMeasurement &measurement,
                                          double residual, int branch) {
      single_anchor_last_valid_ = true;
      single_anchor_last_anchor_id_ = measurement.anchor_id;
      single_anchor_last_branch_ = branch;
      single_anchor_last_range_m_ = measurement.range_m;
      single_anchor_last_residual_m_ = residual;
      single_anchor_last_stamp_ = measurement.stamp;
      single_anchor_confirm_counter_log = single_anchor_confirm_counter_;
    };
    auto rememberCorridorState = [&](double residual) {
      corridor_residual_stable_count_ = corridorStableCountForResidual(residual);
      corridor_last_residual_ = residual;
      corridor_last_residual_valid_ = true;
      corridor_last_tag_position_w_ = tag_position_w;
      corridor_last_tag_valid_ = true;
    };
    auto rememberCorridorDirectionSample = [&]() {
      corridor_direction_history_.push_back(tag_position_w);
      while (corridor_direction_history_.size() > static_cast<size_t>(direction_check_window_frames_))
      {
        corridor_direction_history_.pop_front();
      }
    };
    auto updateCorridorDirectionAngle = [&]() {
      if (corridor_direction_history_.size() < 2) return false;
      const V3D motion = tag_position_w - corridor_direction_history_.front();
      const double motion_xy_norm = std::hypot(motion.x(), motion.y());
      if (motion_xy_norm < min_motion_for_direction_check_m_) return false;
      double cos_angle =
          std::fabs((motion.x() * baseline_direction.x() + motion.y() * baseline_direction.y()) /
                    std::max(motion_xy_norm, 1e-9));
      cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
      corridor_direction_angle_deg = std::acos(cos_angle) * 180.0 / M_PI;
      return true;
    };
    auto recordTwoAnchorReject = [&]() {
      two_anchor_reject_count_++;
      two_anchor_baseline_consistency_sum_ += baseline_consistency_error;
    };
    auto recordTwoAnchorUpdate = [&](double xy_correction_applied) {
      two_anchor_update_count_++;
      two_anchor_xy_correction_sum_ += xy_correction_applied;
      two_anchor_xy_correction_max_ =
          std::max(two_anchor_xy_correction_max_, xy_correction_applied);
      two_anchor_abs_baseline_residual_sum_ += std::fabs(baseline_residual_after_gate);
      two_anchor_baseline_consistency_sum_ += baseline_consistency_error;
    };

    const bool dry_run = residual_debug_only_ || !update_en_;
    if (dry_run)
    {
      skip_reason = residual_debug_only_ ? "two_anchor_policy_dry_run" : "two_anchor_policy_disable";
      log_action("INFO", "dry_run", 0.0, 1.0, V3D::Zero());
      return finalize_result("dry_run", false, 0.0, V3D::Zero());
    }

    if (max_time_diff_s_ > 0.0 && max_abs_time_diff > max_time_diff_s_)
    {
      relocalization_candidate_count_ = 0;
      skip_reason = "time_mismatch";
      log_action("WARN", "skip_time_mismatch", 0.0, 1.0, V3D::Zero());
      return finalize_result("skip_time_mismatch", false, 0.0, V3D::Zero());
    }

    std::string state_machine_update_action;
    if (two_anchor_case)
    {
      if (total_configured_anchors >= 3)
      {
        selected_update_policy = two_anchor_policy_when_total_anchors_gt2_;
        update_mode_for_log = "two_anchor_gt2_policy";
        if (two_anchor_policy_when_total_anchors_gt2_ == "disable")
        {
          skip_reason = "two_anchor_policy_disable";
          log_action("WARN", "skip_not_enough_anchors", 0.0, 1.0, V3D::Zero());
          return finalize_result("skip_not_enough_anchors", false, 0.0, V3D::Zero());
        }
        if (two_anchor_policy_when_total_anchors_gt2_ == "dry_run")
        {
          skip_reason = "two_anchor_policy_dry_run";
          log_action("INFO", "two_anchor_dry_run", 0.0, 1.0, V3D::Zero());
          return finalize_result("two_anchor_dry_run", false, 0.0, V3D::Zero());
        }
        if (two_anchor_policy_when_total_anchors_gt2_ == "weak_xy")
        {
          state_machine_update_action = "two_anchor_weak_xy_update";
        }
        else if (two_anchor_policy_when_total_anchors_gt2_ == "baseline_1d_only_if_pair_matches_corridor")
        {
          if (!baseline_initialized_for_update || !baseline_direction_valid || !two_anchor_uses_baseline_pair)
          {
            skip_reason = !baseline_initialized_for_update ? "baseline_not_initialized" : "anchor_not_on_baseline";
            baseline_residual_after_gate = 0.0;
            recordTwoAnchorReject();
            log_action("WARN", "reject_uwb_outlier", 0.0, 1.0, V3D::Zero());
            return finalize_result("reject_uwb_outlier", false, 0.0, V3D::Zero());
          }
        }
      }

      if (state_machine_update_action.empty() &&
          (two_anchor_update_disabled || two_anchor_update_mode_ == "dry_run"))
      {
        log_action("INFO", "two_anchor_dry_run", 0.0, 1.0, V3D::Zero());
        return finalize_result("two_anchor_dry_run", false, 0.0, V3D::Zero());
      }
      if (state_machine_update_action.empty() &&
          (!baseline_initialized_for_update || !baseline_direction_valid || !two_anchor_uses_baseline_pair))
      {
        skip_reason = !baseline_initialized_for_update ? "baseline_not_initialized" : "anchor_not_on_baseline";
        baseline_residual_after_gate = 0.0;
        recordTwoAnchorReject();
        log_action("WARN", "reject_uwb_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_uwb_outlier", false, 0.0, V3D::Zero());
      }
      if (state_machine_update_action.empty() &&
          two_anchor_baseline_consistency_threshold_m_ > 0.0 &&
          baseline_consistency_error > two_anchor_baseline_consistency_threshold_m_)
      {
        baseline_residual_after_gate = 0.0;
        skip_reason = "baseline_consistency_error_large";
        recordTwoAnchorReject();
        log_action("WARN", "reject_uwb_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_uwb_outlier", false, 0.0, V3D::Zero());
      }
      if (state_machine_update_action.empty() &&
          ((two_anchor_max_residual_rms_ > 0.0 && residual_rms > two_anchor_max_residual_rms_) ||
           (two_anchor_max_abs_residual_ > 0.0 && max_abs_residual > two_anchor_max_abs_residual_)))
      {
        baseline_residual_after_gate = 0.0;
        skip_reason = "two_anchor_residual_gate";
        recordTwoAnchorReject();
        log_action("WARN", "reject_uwb_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_uwb_outlier", false, 0.0, V3D::Zero());
      }
      if (state_machine_update_action.empty())
      {
        baseline_residual_after_gate = baseline_residual;
        if (two_anchor_update_mode_ == "baseline_1d")
        {
          state_machine_update_action = "two_anchor_baseline_1d_update";
        }
        else if (two_anchor_update_mode_ == "baseline_1d_direct")
        {
          state_machine_update_action = "two_anchor_baseline_1d_direct_update";
        }
        else
        {
          state_machine_update_action = "two_anchor_weak_xy_update";
        }
      }
    }
    else if (single_anchor_case)
    {
      if (!single_anchor_corridor_1d_en_)
      {
        log_action("WARN", "skip_not_enough_anchors", 0.0, 1.0, V3D::Zero());
        return finalize_result("skip_not_enough_anchors", false, 0.0, V3D::Zero());
      }
      if (single_anchor_only_when_total_anchors_eq_2_ && total_configured_anchors != 2)
      {
        skip_reason = "total_configured_anchors_not_2";
        log_action("WARN", "reject_single_anchor_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_single_anchor_outlier", false, 0.0, V3D::Zero());
      }
      if (!baseline_initialized_for_update || !baseline_direction_valid)
      {
        skip_reason = "baseline_not_initialized";
        log_action("WARN", "skip_not_enough_anchors", 0.0, 1.0, V3D::Zero());
        return finalize_result("skip_not_enough_anchors", false, 0.0, V3D::Zero());
      }

      const UwbRangeMeasurement &measurement = used_measurements.front();
      single_anchor_id = measurement.anchor_id;
      single_anchor_measured_range = measurement.range_m;
      const bool belongs_to_baseline =
          single_anchor_id == baseline_anchor_start_id_ || single_anchor_id == baseline_anchor_end_id_;
      if (!belongs_to_baseline)
      {
        skip_reason = "anchor_not_on_baseline";
        log_action("WARN", "reject_single_anchor_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_single_anchor_outlier", false, 0.0, V3D::Zero());
      }
      if (measurement.range_m < single_anchor_min_range_m_ ||
          measurement.range_m > single_anchor_max_range_m_)
      {
        skip_reason = "range_out_of_bounds";
        log_action("WARN", "reject_single_anchor_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_single_anchor_outlier", false, 0.0, V3D::Zero());
      }

      const auto anchor_it = anchors_.find(single_anchor_id);
      if (anchor_it == anchors_.end())
      {
        skip_reason = "anchor_position_missing";
        log_action("WARN", "reject_single_anchor_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_single_anchor_outlier", false, 0.0, V3D::Zero());
      }

      baseline_s_pred = (tag_position_w - baseline_start_position).dot(baseline_direction);
      single_anchor_s_anchor = (anchor_it->second.position_w - baseline_start_position).dot(baseline_direction);
      single_anchor_abs_s_pred_minus_s_anchor = std::fabs(baseline_s_pred - single_anchor_s_anchor);
      single_anchor_height_diff = tag_position_w.z() - anchor_it->second.position_w.z();
      const double range_sq =
          measurement.range_m * measurement.range_m - single_anchor_height_diff * single_anchor_height_diff;
      single_anchor_rho = std::sqrt(std::max(0.0, range_sq));
      single_anchor_candidate_1 = single_anchor_s_anchor + single_anchor_rho;
      single_anchor_candidate_2 = single_anchor_s_anchor - single_anchor_rho;
      const double dist1 = std::fabs(single_anchor_candidate_1 - baseline_s_pred);
      const double dist2 = std::fabs(single_anchor_candidate_2 - baseline_s_pred);
      single_anchor_branch_margin = std::fabs(dist1 - dist2);
      selected_max_step = single_anchor_normal_max_step_m_;
      rememberCorridorDirectionSample();
      if (single_anchor_abs_s_pred_minus_s_anchor < single_anchor_near_anchor_disable_dist_m_)
      {
        single_anchor_confirm_counter_ = 0;
        single_anchor_confirm_counter_log = 0;
        skip_reason = "near_anchor";
        log_action("WARN", "skip_single_anchor_near_anchor", 0.0, 1.0, V3D::Zero());
        return finalize_result("skip_single_anchor_near_anchor", false, 0.0, V3D::Zero());
      }
      if (single_anchor_branch_margin < single_anchor_branch_margin_m_)
      {
        single_anchor_confirm_counter_ = 0;
        single_anchor_confirm_counter_log = 0;
        skip_reason = "branch_ambiguous";
        log_action("WARN", "skip_single_anchor_branch_ambiguous", 0.0, 1.0, V3D::Zero());
        return finalize_result("skip_single_anchor_branch_ambiguous", false, 0.0, V3D::Zero());
      }
      if (dist1 <= dist2)
      {
        baseline_s_meas = single_anchor_candidate_1;
        single_anchor_selected_branch = 1;
      }
      else
      {
        baseline_s_meas = single_anchor_candidate_2;
        single_anchor_selected_branch = -1;
      }
      single_anchor_residual = baseline_s_meas - baseline_s_pred;
      baseline_residual = single_anchor_residual;
      baseline_residual_after_gate = single_anchor_residual;

      if (single_anchor_last_valid_ && single_anchor_last_anchor_id_ == single_anchor_id)
      {
        single_anchor_range_jump = std::fabs(measurement.range_m - single_anchor_last_range_m_);
        single_anchor_residual_jump = std::fabs(single_anchor_residual - single_anchor_last_residual_m_);
        const double dt = measurement.stamp - single_anchor_last_stamp_;
        if (dt > 1e-6) single_anchor_estimated_range_speed = single_anchor_range_jump / dt;
      }
      if (single_anchor_range_jump_threshold_m_ > 0.0 &&
          single_anchor_range_jump > single_anchor_range_jump_threshold_m_)
      {
        single_anchor_confirm_counter_ = 0;
        single_anchor_confirm_counter_log = 0;
        skip_reason = "range_jump";
        log_action("WARN", "reject_single_anchor_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_single_anchor_outlier", false, 0.0, V3D::Zero());
      }
      if (single_anchor_speed_threshold_mps_ > 0.0 &&
          single_anchor_estimated_range_speed > single_anchor_speed_threshold_mps_)
      {
        single_anchor_confirm_counter_ = 0;
        single_anchor_confirm_counter_log = 0;
        skip_reason = "range_speed";
        log_action("WARN", "reject_single_anchor_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_single_anchor_outlier", false, 0.0, V3D::Zero());
      }
      if (single_anchor_residual_jump_threshold_m_ > 0.0 &&
          single_anchor_residual_jump > single_anchor_residual_jump_threshold_m_)
      {
        single_anchor_confirm_counter_ = 0;
        single_anchor_confirm_counter_log = 0;
        skip_reason = "residual_jump";
        log_action("WARN", "reject_single_anchor_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_single_anchor_outlier", false, 0.0, V3D::Zero());
      }

      const int stable_count_for_residual = corridorStableCountForResidual(single_anchor_residual);
      if (single_anchor_max_residual_ > 0.0 &&
          std::fabs(single_anchor_residual) > single_anchor_max_residual_)
      {
        skip_reason = "single_anchor_residual_large";
        log_action(stable_count_for_residual >= degraded_confirm_count_ ? "WARN" : "WARN",
                   stable_count_for_residual >= degraded_confirm_count_ ? "hold_large_correction" :
                                                                         "reject_single_anchor_outlier",
                   0.0, 1.0, V3D::Zero());
        return finalize_result(stable_count_for_residual >= degraded_confirm_count_ ? "hold_large_correction" :
                                                                                      "reject_single_anchor_outlier",
                               false, 0.0, V3D::Zero());
      }
      const bool has_corridor_direction_check = updateCorridorDirectionAngle();
      if (disable_single_anchor_on_turn_ &&
          has_corridor_direction_check &&
          corridor_direction_angle_deg > corridor_direction_max_angle_deg_)
      {
        single_anchor_confirm_counter_ = 0;
        single_anchor_confirm_counter_log = 0;
        skip_reason = "turn_or_corner";
        corridor_last_tag_position_w_ = tag_position_w;
        corridor_last_tag_valid_ = true;
        log_action("WARN", "skip_single_anchor_turn_or_corner", 0.0, 1.0, V3D::Zero());
        return finalize_result("skip_single_anchor_turn_or_corner", false, 0.0, V3D::Zero());
      }

      if (single_anchor_last_valid_ &&
          single_anchor_last_anchor_id_ == single_anchor_id &&
          single_anchor_last_branch_ == single_anchor_selected_branch)
      {
        single_anchor_confirm_counter_++;
      }
      else
      {
        single_anchor_confirm_counter_ = 1;
      }
      single_anchor_confirm_counter_log = single_anchor_confirm_counter_;
      rememberSingleAnchorSample(measurement, single_anchor_residual, single_anchor_selected_branch);
      if (single_anchor_confirm_counter_ < single_anchor_confirm_count_required_)
      {
        skip_reason = "wait_confirm";
        reason_not_confirmed = "wait_single_anchor_confirm";
        log_action("INFO", "wait_single_anchor_confirm", 0.0, 1.0, V3D::Zero());
        return finalize_result("wait_single_anchor_confirm", false, 0.0, V3D::Zero());
      }

      selected_max_step = selectCorridorMaxStep(true, stable_count_for_residual);
      const double direct_step_raw = single_anchor_alpha_ * single_anchor_residual;
      double clamp_ratio = 1.0;
      if (selected_max_step > 0.0 && std::fabs(direct_step_raw) > selected_max_step)
      {
        clamp_ratio = selected_max_step / std::max(std::fabs(direct_step_raw), 1e-9);
      }
      delta_s = direct_step_raw * clamp_ratio;

      dx_raw.setZero();
      dx_raw(3) = direct_step_raw * baseline_direction.x();
      dx_raw(4) = direct_step_raw * baseline_direction.y();
      dx_after_baseline_projection = dx_raw;

      H = Eigen::MatrixXd::Zero(1, DIM_STATE);
      H(0, 3) = baseline_direction.x();
      H(0, 4) = baseline_direction.y();
      z = Eigen::VectorXd::Constant(1, single_anchor_residual);
      Eigen::MatrixXd K_used = Eigen::MatrixXd::Zero(DIM_STATE, 1);
      K_used(3, 0) = single_anchor_alpha_ * clamp_ratio * baseline_direction.x();
      K_used(4, 0) = single_anchor_alpha_ * clamp_ratio * baseline_direction.y();
      kalman_gain_norm = K_used.norm();
      const Eigen::VectorXd dx_dynamic = K_used * z;
      VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
      dx = dx_dynamic;
      dx_after_clamp = dx;

      const V3D trans_add = dx.block<3, 1>(3, 0);
      xy_correction_raw = std::hypot(dx_raw(3), dx_raw(4));
      result.xy_correction_raw = xy_correction_raw;
      const double xy_correction_applied = std::hypot(trans_add.x(), trans_add.y());

      Eigen::MatrixXd covariance_updated;
      double covariance_asymmetry = 0.0;
      double covariance_min_diagonal = 0.0;
      if (!computeUwbJosephCovariance(cov_for_uwb, H, R, K_used, covariance_updated,
                                      covariance_asymmetry, covariance_min_diagonal))
      {
        skip_reason = "covariance_invalid";
        log_action("WARN", "reject_covariance_invalid", 0.0, 1.0, V3D::Zero());
        report.primary_reason = UwbRejectReason::COVARIANCE_INVALID;
        report.outcome = UwbUpdateOutcome::REJECTED;
        return finalize_result("reject_covariance_invalid", false, 0.0, V3D::Zero());
      }
      state += dx;
      state.cov = covariance_updated;
      snapStateForDeterminism(state);
      position_cov_after_update = state.cov.block<3, 3>(3, 3).diagonal();
      logCovarianceStages("single_anchor_corridor_1d_update", covariance_updated,
                          covariance_asymmetry, covariance_min_diagonal);
      rememberCorridorState(single_anchor_residual);

      log_action("INFO", "single_anchor_corridor_1d_update",
                 xy_correction_applied, clamp_ratio, trans_add);
      return finalize_result("single_anchor_corridor_1d_update", true,
                             xy_correction_applied, trans_add);
    }
    else if (used_anchor_count >= 3)
    {
      update_mode_for_log = "multi_anchor";
      selected_update_policy = "multi_anchor_xy_update_limited";
      const bool geometry_good = uwb_only_valid && uwb_only_geometry_score >= anchor_geometry_min_score_;
      const bool multi_anchor_normal =
          residual_rms < multi_anchor_max_residual_rms_ &&
          max_abs_residual < multi_anchor_max_abs_residual_ &&
          xy_correction_raw < normal_update_max_xy_raw_ &&
          geometry_good;
      const bool debug_force_limited_update =
          uwb_debug_force_limited_update_ &&
          residual_rms < limited_update_max_residual_rms_ &&
          max_abs_residual < limited_update_max_abs_residual_ &&
          xy_correction_raw < limited_update_max_xy_raw_ &&
          (max_time_diff_s_ <= 0.0 || max_abs_time_diff <= max_time_diff_s_) &&
          geometry_good;
      const bool slam_uwb_residual_large =
          residual_rms >= multi_anchor_max_residual_rms_ ||
          max_abs_residual >= multi_anchor_max_abs_residual_;
      const bool uwb_only_good =
          uwb_only_valid &&
          result.uwb_only_residual_rms < uwb_only_max_residual_rms_ &&
          result.uwb_only_max_abs_residual < uwb_only_max_abs_residual_ &&
          result.uwb_only_position_jump <= uwb_position_jump_threshold_m_ &&
          result.uwb_only_speed <= uwb_speed_threshold_mps_ &&
          geometry_good;
      const bool relocalization_candidate =
          slam_uwb_residual_large &&
          uwb_only_good &&
          result.slam_uwb_position_diff > relocalization_threshold_m_;

      if (multi_anchor_normal || debug_force_limited_update)
      {
        relocalization_candidate_count_ = 0;
        state_machine_update_action = "multi_anchor_xy_update_limited";
        if (debug_force_limited_update && !multi_anchor_normal)
        {
          selected_update_policy = "debug_force_limited_update";
          skip_reason = "debug_force_limited_update";
        }
      }
      else if (relocalization_candidate)
      {
        relocalization_candidate_count_total_++;
        relocalization_slam_diff_sum_ += result.slam_uwb_position_diff;
        relocalization_uwb_only_residual_sum_ += result.uwb_only_residual_rms;
        relocalization_uwb_only_jump_sum_ += result.uwb_only_position_jump;
        relocalization_candidate_count_++;
        result.relocalization_candidate_count = relocalization_candidate_count_;
        if (relocalization_candidate_count_ >= relocalization_confirm_count_)
        {
          result.relocalization_confirmed = relocalization_en_;
          reason_not_confirmed = relocalization_en_ ? "none" : "relocalization_disabled";
          const std::string action = relocalization_en_ ? "relocalization_confirmed" : "relocalization_candidate";
          log_action("WARN", action, 0.0, 1.0, V3D::Zero());
          return finalize_result(action, false, 0.0, V3D::Zero(), relocalization_en_);
        }
        reason_not_confirmed = "waiting_relocalization_confirm";
        log_action("WARN", "relocalization_candidate", 0.0, 1.0, V3D::Zero());
        return finalize_result("relocalization_candidate", false, 0.0, V3D::Zero(), false);
      }
      else
      {
        relocalization_candidate_count_ = 0;
        if (!uwb_only_valid)
        {
          skip_reason = "uwb_only_solve_failed";
          reason_not_confirmed = "uwb_only_solve_failed";
        }
        else if (!geometry_good)
        {
          skip_reason = "bad_anchor_geometry";
          reason_not_confirmed = "bad_anchor_geometry";
        }
        else if (xy_correction_raw >= limited_update_max_xy_raw_)
        {
          skip_reason = "xy_correction_raw_too_large";
          reason_not_confirmed = "xy_correction_raw_too_large";
        }
        else if (residual_rms >= multi_anchor_max_residual_rms_ ||
                 max_abs_residual >= multi_anchor_max_abs_residual_)
        {
          skip_reason = "multi_anchor_residual_gate";
          reason_not_confirmed = !uwb_only_good ? "uwb_only_not_stable" : "slam_uwb_diff_below_threshold";
        }
        else if (xy_correction_raw >= normal_update_max_xy_raw_)
        {
          skip_reason = "multi_anchor_xy_raw_gate";
          reason_not_confirmed = "xy_raw_above_normal_gate";
        }
        else
        {
          skip_reason = "multi_anchor_gate";
          reason_not_confirmed = "multi_anchor_gate";
        }
        log_action("WARN", "reject_uwb_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_uwb_outlier", false, 0.0, V3D::Zero());
      }
    }

    bool allow_limited_large_correction = false;
    if (state_machine_update_action.empty())
    {
    const double normal_residual_rms = max_residual_rms_ > 0.0 ? max_residual_rms_ : 0.5;
    constexpr double lost_residual_rms = 2.0;
    constexpr double relocalize_residual_rms = 3.0;
    constexpr int lost_recovery_good_updates = 5;

    if (two_anchor_update_disabled)
    {
      log_action("INFO", "two_anchor_dry_run", 0.0, 1.0, V3D::Zero());
      return finalize_result("two_anchor_dry_run", false, 0.0, V3D::Zero());
    }

    if (two_anchor_case && two_anchor_update_mode_ == "dry_run")
    {
      log_action("INFO", "two_anchor_dry_run", 0.0, 1.0, V3D::Zero());
      return finalize_result("two_anchor_dry_run", false, 0.0, V3D::Zero());
    }

    if (two_anchor_case &&
        two_anchor_baseline_consistency_threshold_m_ > 0.0 &&
        baseline_consistency_error > two_anchor_baseline_consistency_threshold_m_)
    {
      skip_reason = "baseline_consistency_error_large";
      recordTwoAnchorReject();
      log_action("WARN", "two_anchor_baseline_inconsistent", 0.0, 1.0, V3D::Zero());
      return finalize_result("two_anchor_baseline_inconsistent", false, 0.0, V3D::Zero());
    }

    if (residual_rms >= relocalize_residual_rms)
    {
      uwb_state_ = 2;
      uwb_consecutive_good_count_ = 0;
      uwb_lost_good_count_ = 0;
      uwb_consecutive_gate_ready_ = false;
      skip_reason = "residual_rms_relocalize_gate";
      log_action("WARN", "need_relocalize_or_hold", 0.0, 1.0, V3D::Zero());
      return finalize_result("need_relocalize_or_hold", false, 0.0, V3D::Zero(), true);
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
        reason_not_confirmed = "lost_recovered_wait";
        log_action("INFO", "lost_recovered_wait", 0.0, 1.0, V3D::Zero());
        return finalize_result("lost_recovered_wait", false, 0.0, V3D::Zero());
      }
      else
      {
        skip_reason = "lost_hold";
        log_action("WARN", "lost_hold", 0.0, 1.0, V3D::Zero());
        return finalize_result("lost_hold", false, 0.0, V3D::Zero());
      }
    }

    if (residual_rms >= lost_residual_rms)
    {
      uwb_state_ = 2;
      uwb_consecutive_good_count_ = 0;
      uwb_lost_good_count_ = 0;
      if (lost_hold_en_)
      {
        skip_reason = "lost_hold";
        log_action("WARN", "lost_hold", 0.0, 1.0, V3D::Zero());
        return finalize_result("lost_hold", false, 0.0, V3D::Zero());
      }
    }
    else if (residual_rms >= normal_residual_rms)
    {
      uwb_state_ = 1;
      uwb_consecutive_good_count_ = 0;
      uwb_lost_good_count_ = 0;
      if (suspect_hold_en_)
      {
        skip_reason = "suspect_hold";
        log_action("WARN", "suspect_hold", 0.0, 1.0, V3D::Zero());
        return finalize_result("suspect_hold", false, 0.0, V3D::Zero());
      }
    }
    else
    {
      uwb_state_ = 0;
      uwb_lost_good_count_ = 0;
    }

    if (hard_reject_xy_raw_ > 0.0 && xy_correction_raw > hard_reject_xy_raw_)
    {
      limited_update_consecutive_good_count_ = 0;
      skip_reason = "hard_reject_xy_raw";
      log_action("WARN", "hard_reject_xy_raw", 0.0, 1.0, V3D::Zero());
      return finalize_result("hard_reject_xy_raw", false, 0.0, V3D::Zero());
    }

    const bool large_correction = xy_correction_raw > normal_update_max_xy_raw_;
    const bool limited_update_stable =
        large_correction &&
        used_anchor_count >= 2 &&
        residual_rms < limited_update_max_residual_rms_ &&
        max_abs_residual < limited_update_max_abs_residual_ &&
        max_abs_time_diff <= limited_update_max_time_diff_s_ &&
        xy_correction_raw < limited_update_max_xy_raw_;
    if (limited_update_stable)
    {
      limited_update_consecutive_good_count_++;
    }
    else
    {
      limited_update_consecutive_good_count_ = 0;
    }
    result.limited_update_consecutive_good_count = limited_update_consecutive_good_count_;
    allow_limited_large_correction =
        limited_update_stable &&
        limited_update_consecutive_good_count_ >= limited_update_require_consecutive_good_;
    if (large_correction && !limited_update_stable)
    {
      uwb_consecutive_good_count_ = 0;
      skip_reason = "large_correction_not_stable";
      log_action("WARN", "large_correction_hold", 0.0, 1.0, V3D::Zero());
      return finalize_result("large_correction_hold", false, 0.0, V3D::Zero());
    }
    if (large_correction && !allow_limited_large_correction)
    {
      reason_not_confirmed = "wait_limited_consecutive_good";
      log_action("INFO", "wait_limited_consecutive_good", 0.0, 1.0, V3D::Zero());
      return finalize_result("wait_limited_consecutive_good", false, 0.0, V3D::Zero());
    }
    if (allow_limited_large_correction &&
        relocalization_candidate_min_xy_raw_ > 0.0 &&
        xy_correction_raw >= relocalization_candidate_min_xy_raw_)
    {
      relocalization_candidate_count_total_++;
      relocalization_slam_diff_sum_ += result.slam_uwb_position_diff;
      relocalization_uwb_only_residual_sum_ += result.uwb_only_residual_rms;
      relocalization_uwb_only_jump_sum_ += result.uwb_only_position_jump;
      relocalization_candidate_count_++;
      result.relocalization_candidate_count = relocalization_candidate_count_;
      const bool confirmed =
          relocalization_en_ && relocalization_candidate_count_ >= relocalization_confirm_count_;
      result.relocalization_confirmed = confirmed;
      reason_not_confirmed = confirmed ? "none" :
          (relocalization_en_ ? "waiting_relocalization_confirm" : "relocalization_disabled");
      const std::string action = confirmed ? "relocalization_confirmed" : "relocalization_candidate";
      log_action("WARN", action, 0.0, 1.0, V3D::Zero());
      return finalize_result(action, false, 0.0, V3D::Zero(), confirmed);
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
        reason_not_confirmed = "wait_consecutive_good";
        log_action("INFO", "wait_consecutive_good", 0.0, 1.0, V3D::Zero());
        return finalize_result("wait_consecutive_good", false, 0.0, V3D::Zero());
      }
      uwb_consecutive_gate_ready_ = true;
    }
    }

    if (state_machine_update_action == "two_anchor_baseline_1d_direct_update")
    {
      if (!baseline_direction_valid)
      {
        skip_reason = "baseline_direction_invalid";
        recordTwoAnchorReject();
        log_action("WARN", "reject_uwb_outlier", 0.0, 1.0, V3D::Zero());
        return finalize_result("reject_uwb_outlier", false, 0.0, V3D::Zero());
      }

      const int stable_count_for_residual = corridorStableCountForResidual(baseline_residual_after_gate);
      if (two_anchor_max_residual_ > 0.0 &&
          std::fabs(baseline_residual_after_gate) > two_anchor_max_residual_)
      {
        skip_reason = "two_anchor_residual_large";
        const bool stable_large = stable_count_for_residual >= degraded_confirm_count_;
        recordTwoAnchorReject();
        log_action("WARN", stable_large ? "hold_large_correction" : "reject_uwb_outlier",
                   0.0, 1.0, V3D::Zero());
        return finalize_result(stable_large ? "hold_large_correction" : "reject_uwb_outlier",
                               false, 0.0, V3D::Zero());
      }

      selected_max_step = selectCorridorMaxStep(false, stable_count_for_residual);
      const double direct_step_raw = baseline_1d_direct_alpha_ * baseline_residual_after_gate;
      double clamp_ratio = 1.0;
      if (selected_max_step > 0.0 && std::fabs(direct_step_raw) > selected_max_step)
      {
        clamp_ratio = selected_max_step / std::max(std::fabs(direct_step_raw), 1e-9);
      }
      delta_s = direct_step_raw * clamp_ratio;

      dx_raw.setZero();
      dx_raw(3) = direct_step_raw * baseline_direction.x();
      dx_raw(4) = direct_step_raw * baseline_direction.y();
      dx_after_baseline_projection = dx_raw;

      Eigen::MatrixXd K_used = Eigen::MatrixXd::Zero(DIM_STATE, 1);
      K_used(3, 0) = baseline_1d_direct_alpha_ * clamp_ratio * baseline_direction.x();
      K_used(4, 0) = baseline_1d_direct_alpha_ * clamp_ratio * baseline_direction.y();
      // ponytail: the direct alpha is the estimator gain here, so Joseph's K*R*K' term can raise covariance.
      kalman_gain_norm = K_used.norm();
      const Eigen::VectorXd dx_dynamic = K_used * z;
      VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
      dx = dx_dynamic;
      dx_after_clamp = dx;

      const V3D rot_add = V3D::Zero();
      const V3D trans_add = dx.block<3, 1>(3, 0);
      xy_correction_raw = std::hypot(dx_raw(3), dx_raw(4));
      result.xy_correction_raw = xy_correction_raw;
      const double xy_correction_applied = std::hypot(trans_add.x(), trans_add.y());

      Eigen::MatrixXd covariance_updated;
      double covariance_asymmetry = 0.0;
      double covariance_min_diagonal = 0.0;
      if (!computeUwbJosephCovariance(cov_for_uwb, H, R, K_used, covariance_updated,
                                      covariance_asymmetry, covariance_min_diagonal))
      {
        skip_reason = "covariance_invalid";
        recordTwoAnchorReject();
        log_action("WARN", "reject_covariance_invalid", 0.0, 1.0, V3D::Zero());
        report.primary_reason = UwbRejectReason::COVARIANCE_INVALID;
        report.outcome = UwbUpdateOutcome::REJECTED;
        return finalize_result("reject_covariance_invalid", false, 0.0, V3D::Zero());
      }
      state += dx;
      state.cov = covariance_updated;
      snapStateForDeterminism(state);
      position_cov_after_update = state.cov.block<3, 3>(3, 3).diagonal();
      logCovarianceStages(state_machine_update_action, covariance_updated,
                          covariance_asymmetry, covariance_min_diagonal);
      rememberCorridorState(baseline_residual_after_gate);
      recordTwoAnchorUpdate(xy_correction_applied);

      log_action("INFO", state_machine_update_action,
                 xy_correction_applied, clamp_ratio, trans_add);
      return finalize_result(state_machine_update_action, true, xy_correction_applied, trans_add);
    }

    double clamp_ratio = 1.0;
    if (max_update_step_xy_ > 0.0 && xy_correction_raw > max_update_step_xy_)
    {
      clamp_ratio = max_update_step_xy_ / std::max(xy_correction_raw, 1e-9);
    }

    const Eigen::MatrixXd K_used = K_constrained * clamp_ratio;
    kalman_gain_norm = K_used.norm();
    Eigen::VectorXd dx_dynamic = K_used * z;
    if (dx_dynamic.size() != DIM_STATE || !dx_dynamic.allFinite())
    {
      result.action = "skip_non_finite_dx";
      report.action = result.action;
      report.outcome = UwbUpdateOutcome::REJECTED;
      report.primary_reason = UwbRejectReason::NON_FINITE_CORRECTION;
      return result;
    }

    VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
    dx = dx_dynamic;
    dx_after_clamp = dx;

    V3D rot_add = dx.block<3, 1>(0, 0);
    V3D trans_add = dx.block<3, 1>(3, 0);
    const double xy_correction_applied = std::hypot(trans_add.x(), trans_add.y());

    Eigen::MatrixXd covariance_updated;
    double covariance_asymmetry = 0.0;
    double covariance_min_diagonal = 0.0;
    if (!computeUwbJosephCovariance(cov_for_uwb, H, R, K_used, covariance_updated,
                                    covariance_asymmetry, covariance_min_diagonal))
    {
      skip_reason = "covariance_invalid";
      log_action("WARN", "reject_covariance_invalid", 0.0, 1.0, V3D::Zero());
      report.primary_reason = UwbRejectReason::COVARIANCE_INVALID;
      report.outcome = UwbUpdateOutcome::REJECTED;
      return finalize_result("reject_covariance_invalid", false, 0.0, V3D::Zero());
    }
    state += dx;
    state.cov = covariance_updated;
    snapStateForDeterminism(state);
    position_cov_after_update = state.cov.block<3, 3>(3, 3).diagonal();
    const std::string update_action = !state_machine_update_action.empty() ? state_machine_update_action :
                                      (allow_limited_large_correction ? "xy_update_limited" :
                                       (two_anchor_case ? "xy_update_weak_2anchors" : "xy_update"));
    logCovarianceStages(update_action, covariance_updated,
                        covariance_asymmetry, covariance_min_diagonal);
    log_action("INFO", update_action,
               xy_correction_applied, clamp_ratio, trans_add);
    return finalize_result(update_action, true, xy_correction_applied, trans_add);
  }

  if (residual_debug_only_ || !update_en_)
  {
    result.used_count = used_anchor_count;
    result.action = "dry_run";
    result.residual_rms = residual_rms;
    result.max_abs_residual = max_abs_residual;
    result.time_diff = time_diff_for_log;
    report.action = result.action;
    report.mode = "joint_tag_offset";
    report.outcome = UwbUpdateOutcome::SKIPPED;
    report.primary_reason = residual_debug_only_ ? UwbRejectReason::DEBUG_ONLY :
                                                   UwbRejectReason::UPDATE_DISABLED;
    report.residual_rms = residual_rms;
    report.max_abs_residual = max_abs_residual;
    return result;
  }

  constexpr int DIM_UWB_JOINT = DIM_STATE + 3;
  Eigen::MatrixXd H_joint = Eigen::MatrixXd::Zero(row, DIM_UWB_JOINT);
  H_joint.block(0, 0, row, DIM_STATE) = H;
  H_joint.block(0, DIM_STATE, row, 3) = H_tag;

  Eigen::MatrixXd P_joint = Eigen::MatrixXd::Zero(DIM_UWB_JOINT, DIM_UWB_JOINT);
  P_joint.block(0, 0, DIM_STATE, DIM_STATE) = state.cov;
  const V3D position_cov_before_update = state.cov.block<3, 3>(3, 3).diagonal();
  const double position_cov_floor_used = effectivePositionCovFloor();
  const bool covariance_degraded = degraded_mode_en_ && degraded_mode_;
  bool covariance_inflated = false;
  if (position_cov_floor_used > 0.0)
  {
    const double floor_var = position_cov_floor_used * position_cov_floor_used;
    for (int i = 0; i < 3; ++i)
    {
      const int idx = 3 + i;
      if (P_joint(idx, idx) < floor_var)
      {
        P_joint(idx, idx) = floor_var;
        covariance_inflated = true;
      }
    }
  }
  const double tag_process_var = tag_offset_process_noise_m_ * tag_offset_process_noise_m_;
  P_joint.block(DIM_STATE, DIM_STATE, 3, 3) =
      tag_offset_cov_ + M3D::Identity() * tag_process_var;

  const Eigen::MatrixXd S = H_joint * P_joint * H_joint.transpose() + R;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success)
  {
    result.action = "skip_cov_decomposition";
    report.action = result.action;
    report.mode = "joint_tag_offset";
    report.outcome = UwbUpdateOutcome::REJECTED;
    report.primary_reason = UwbRejectReason::COVARIANCE_INVALID;
    return result;
  }

  const Eigen::MatrixXd K = P_joint * H_joint.transpose() * ldlt.solve(Eigen::MatrixXd::Identity(row, row));
  Eigen::MatrixXd K_constrained = K;
  K_constrained.topRows(DIM_STATE) = applyUwbUpdateMaskAndProjection(
      K.topRows(DIM_STATE), update_z_ && !update_xy_only_,
      update_orientation_ && !update_xy_only_);
  Eigen::VectorXd dx_dynamic = K_constrained * z;
  if (dx_dynamic.size() != DIM_UWB_JOINT || !dx_dynamic.allFinite())
  {
    result.action = "skip_non_finite_dx";
    report.action = result.action;
    report.mode = "joint_tag_offset";
    report.outcome = UwbUpdateOutcome::REJECTED;
    report.primary_reason = UwbRejectReason::NON_FINITE_CORRECTION;
    return result;
  }

  const V3D raw_position_add = dx_dynamic.segment<3>(3);
  const V3D raw_tag_offset_add = dx_dynamic.segment<3>(DIM_STATE);

  V3D rot_add = dx_dynamic.segment<3>(0);
  V3D trans_add = raw_position_add;
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
  if (tag_offset_update_max_step_m_ > 0.0 && raw_tag_offset_add.norm() > tag_offset_update_max_step_m_)
  {
    step_scale = std::min(step_scale, tag_offset_update_max_step_m_ /
                         std::max(raw_tag_offset_add.norm(), 1e-9));
  }
  if (tag_offset_max_norm_m_ > 0.0 &&
      (tag_offset_est_body_ + step_scale * raw_tag_offset_add).norm() > tag_offset_max_norm_m_)
  {
    double low = 0.0;
    double high = step_scale;
    for (int i = 0; i < 40; ++i)
    {
      const double mid = 0.5 * (low + high);
      if ((tag_offset_est_body_ + mid * raw_tag_offset_add).norm() <= tag_offset_max_norm_m_) low = mid;
      else high = mid;
    }
    step_scale = low;
  }

  const Eigen::MatrixXd K_used = K_constrained * step_scale;
  dx_dynamic = K_used * z;
  VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
  dx = dx_dynamic.head(DIM_STATE);
  V3D tag_offset_add = dx_dynamic.segment<3>(DIM_STATE);
  rot_add = dx.block<3, 1>(0, 0);
  trans_add = dx.block<3, 1>(3, 0);

  Eigen::MatrixXd P_joint_updated;
  double covariance_asymmetry = 0.0;
  double covariance_min_diagonal = 0.0;
  if (!computeUwbJosephCovariance(P_joint, H_joint, R, K_used, P_joint_updated,
                                  covariance_asymmetry, covariance_min_diagonal))
  {
    result.action = "reject_covariance_invalid";
    report.action = result.action;
    report.mode = "joint_tag_offset";
    report.outcome = UwbUpdateOutcome::REJECTED;
    report.primary_reason = UwbRejectReason::COVARIANCE_INVALID;
    return result;
  }

  state += dx;
  tag_offset_est_body_ += tag_offset_add;
  state.cov = P_joint_updated.block(0, 0, DIM_STATE, DIM_STATE);
  tag_offset_cov_ = P_joint_updated.block(DIM_STATE, DIM_STATE, 3, 3);
  snapStateForDeterminism(state);

  result.used_count = used_anchor_count;
  result.action = "joint_tag_offset_update";
  result.residual_rms = residual_rms;
  result.max_abs_residual = max_abs_residual;
  result.xy_correction_raw = std::hypot(trans_add.x(), trans_add.y());
  result.xy_correction_applied = result.xy_correction_raw;
  result.time_diff = time_diff_for_log;
  result.correction_norm = trans_add.norm();
  result.baseline_consistency_error = baseline_consistency_error;
  result.state_updated = true;
  result.covariance_inflated = covariance_inflated;
  result.request_pause_map_insert = result.xy_correction_applied > 0.05;
  result.limited_update_consecutive_good_count = limited_update_consecutive_good_count_;
  result.relocalization_candidate_count = relocalization_candidate_count_;
  report.action = result.action;
  report.mode = "joint_tag_offset";
  report.outcome = UwbUpdateOutcome::ACCEPTED;
  report.primary_reason = UwbRejectReason::NONE;
  report.residual_rms = residual_rms;
  report.max_abs_residual = max_abs_residual;
  report.raw_position_correction = raw_position_add;
  report.correction_clamped = step_scale < 1.0 - 1e-12;
  {
    std::ostringstream oss;
    oss << "[UWB_DEBUG_UPDATE] attempt=" << report.attempt_id
        << " action=joint_tag_offset_update raw_delta=(" << raw_position_add.transpose() << ")"
        << " applied_delta=(" << trans_add.transpose() << ")"
        << " tag_offset_add=(" << tag_offset_add.transpose() << ")"
        << " step_scale=" << step_scale
        << " kalman_gain_norm=" << K_used.norm();
    report.deferred_debug_lines.push_back(oss.str());
  }
  {
    const V3D position_cov_after_floor =
        P_joint.block<3, 3>(3, 3).diagonal();
    const V3D position_cov_after_kalman =
        P_joint_updated.block<3, 3>(3, 3).diagonal();
    std::ostringstream oss;
    oss << "[UWB_DEBUG_COVARIANCE] attempt=" << report.attempt_id
        << " update_mode=joint_tag_offset_update"
        << " degradation_level=" << (covariance_degraded ? "degraded" : "normal")
        << " effective_sigma=" << effective_range_noise_m
        << " measurement_variance=" << effective_range_noise_m * effective_range_noise_m
        << " kalman_gain_norm=" << K_used.norm()
        << " position_cov_before=(" << position_cov_before_update.transpose() << ")"
        << " position_cov_after_kalman=(" << position_cov_after_kalman.transpose() << ")"
        << " position_cov_after_floor=(" << position_cov_after_floor.transpose() << ")"
        << " position_cov_after_direct_reset=(" << position_cov_after_kalman.transpose() << ")"
        << " position_cov_final=("
        << state.cov.block<3, 3>(3, 3).diagonal().transpose() << ")"
        << " position_cov_floor_used=" << position_cov_floor_used
        << " covariance_inflated=" << static_cast<int>(covariance_inflated)
        << " covariance_reset=0"
        << " covariance_reset_reason=NONE"
        << " max_asymmetry=" << covariance_asymmetry
        << " min_diagonal=" << covariance_min_diagonal
        << " finite=" << static_cast<int>(state.cov.allFinite());
    report.deferred_debug_lines.push_back(oss.str());
  }

  if (tag_offset_add.norm() <= tag_offset_convergence_step_m_)
    tag_offset_convergence_counter_++;
  else
  {
    tag_offset_convergence_counter_ = 0;
    tag_offset_estimation_ready_ = false;
  }
  if (!tag_offset_estimation_ready_ &&
      tag_offset_convergence_counter_ >= tag_offset_convergence_count_)
  {
    tag_offset_estimation_ready_ = true;
    tag_offset_estimate_version_++;
    std::ostringstream oss;
    oss << "[UWB_TAG_OFFSET_ESTIMATION] status=READY estimate_version="
        << tag_offset_estimate_version_;
    report.deferred_result_lines.push_back(oss.str());
    logFinalTagOffset("online_estimation", &report);
  }
  else if (!tag_offset_estimation_ready_)
  {
    std::ostringstream oss;
    oss << "[UWB_TAG_OFFSET_ESTIMATION] attempt=" << report.attempt_id
        << " status=ESTIMATING stable_count=" << tag_offset_convergence_counter_
        << " required=" << tag_offset_convergence_count_
        << " translation=(" << tag_offset_est_body_.transpose() << ")m";
    report.deferred_debug_lines.push_back(oss.str());
  }
  return result;
}
