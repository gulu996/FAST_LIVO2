#include "rtk_fixed_lag_backend.h"

#include <geometry_msgs/PoseStamped.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <boost/make_shared.hpp>
#include <boost/pointer_cast.hpp>

namespace fast_livo_backend {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr std::size_t kMaximumUnpairedGnssMessages = 500;
constexpr std::size_t kMaximumAlignmentPairs = 2000;
constexpr std::size_t kAlignmentPairsToDropAtCapacity = 1000;

constexpr char kWaitingForRawOdom[] = "WAITING_FOR_RAW_ODOM";
constexpr char kGnssTooOldForBuffer[] = "GNSS_TOO_OLD_FOR_BUFFER";
constexpr char kInterpolationGapTooLarge[] =
    "RAW_ODOM_INTERPOLATION_GAP_TOO_LARGE";
constexpr char kInterpolationInvalid[] = "RAW_ODOM_INTERPOLATION_INVALID";
constexpr char kGnssLateOutOfOrder[] = "GNSS_LATE_OUT_OF_ORDER";
constexpr char kGnssRateLimited[] = "GNSS_RATE_LIMITED";
constexpr char kDuplicateGnssTimestamp[] = "DUPLICATE_GNSS_TIMESTAMP";
constexpr char kNoActiveGraphState[] = "NO_ACTIVE_GRAPH_STATE";
constexpr char kAlignmentTransitionTooOld[] =
    "GNSS_ALIGNMENT_TRANSITION_TOO_OLD";
constexpr char kAlignmentReset[] = "GNSS_ALIGNMENT_RESET";

double clamp(double value, double minimum, double maximum) {
  return std::max(minimum, std::min(value, maximum));
}

double stampDifference(const ros::Time &a, const ros::Time &b) {
  return std::abs((a - b).toSec());
}

std::int64_t stampNanoseconds(const ros::Time &stamp) {
  return static_cast<std::int64_t>(stamp.toNSec());
}

std::string csvField(std::string value) {
  std::replace(value.begin(), value.end(), '\n', ' ');
  std::replace(value.begin(), value.end(), '\r', ' ');
  if (value.find_first_of(",\"") == std::string::npos) return value;
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char character : value) {
    if (character == '"') escaped.push_back('"');
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

}  // namespace

GnssPositionArmFactor::GnssPositionArmFactor(
    gtsam::Key key, const gtsam::Point3 &measurement,
    const gtsam::Point3 &lever_arm,
    const gtsam::SharedNoiseModel &noise_model)
    : Base(noise_model, key),
      measurement_(measurement),
      lever_arm_(lever_arm) {}

gtsam::NonlinearFactor::shared_ptr GnssPositionArmFactor::clone() const {
  return boost::static_pointer_cast<gtsam::NonlinearFactor>(
      boost::make_shared<GnssPositionArmFactor>(*this));
}

gtsam::Vector GnssPositionArmFactor::evaluateError(
    const gtsam::Pose3 &pose,
    boost::optional<gtsam::Matrix &> jacobian) const {
  gtsam::Matrix36 pose_jacobian;
  const gtsam::Point3 predicted = pose.transformFrom(lever_arm_, pose_jacobian);
  if (jacobian) *jacobian = pose_jacobian;
  return predicted - measurement_;
}

RtkFixedLagBackend::RtkFixedLagBackend(ros::NodeHandle &nh) {
  loadParameters(nh);
  validateParameters();
  if (!config_.enable) {
    ROS_INFO("[RTK_BACKEND] Disabled by rtk_backend/enable=false.");
    return;
  }

  bool legacy_gnss_update_enabled = false;
  bool uwb_update_enabled = false;
  nh.param("gps/update_en", legacy_gnss_update_enabled, false);
  nh.param("uwb/update_en", uwb_update_enabled, false);
  if (legacy_gnss_update_enabled || uwb_update_enabled) {
    throw std::invalid_argument(
        "rtk_fixed_lag_backend requires gps/update_en=false and "
        "uwb/update_en=false so /backend/livo_odom_raw stays independent");
  }

  initializeResultFiles();
  setupRos(nh);
  queueTextEvent("BACKEND_INIT", "waiting_for_rtk_fixed_alignment");
  ROS_INFO_STREAM(
      "[RTK_BACKEND] lag=" << config_.lag_seconds
      << "s raw_buffer=" << config_.raw_odom_buffer_seconds
      << "s interpolation_gap_max="
      << config_.max_raw_odom_interpolation_gap_s
      << "s reuse_node_dt=" << config_.reuse_existing_node_time_diff_s
      << "s gnss_node_min_interval=" << config_.gnss_node_min_interval_s
      << "s max_active_states=" << config_.max_active_states
      << " output_directory=" << config_.output_directory);
  ROS_INFO_STREAM(
      "[RTK_BACKEND] LIVO Pose3 sigmas [rotation xyz, translation xyz]=["
      << config_.livo_rotation_sigma_rad << " "
      << config_.livo_rotation_sigma_rad << " "
      << config_.livo_rotation_sigma_rad << " "
      << config_.livo_translation_sigma_m << " "
      << config_.livo_translation_sigma_m << " "
      << config_.livo_translation_sigma_m << "]");
}

RtkFixedLagBackend::~RtkFixedLagBackend() {
  if (!config_.enable) return;
  FileBatch batch;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    archiveActiveStates();
    const std::size_t waiting_graph = pending_factor_gnss_.size();
    const std::size_t waiting_alignment = pending_alignment_gnss_.size();
    const std::size_t waiting_status = pending_status_.size();
    const std::size_t waiting_odom = pending_gnss_odom_.size();
    if (waiting_graph + waiting_alignment + waiting_status + waiting_odom !=
        0) {
      std::ostringstream waiting_detail;
      waiting_detail << "graph=" << waiting_graph
                     << " alignment=" << waiting_alignment
                     << " status=" << waiting_status
                     << " odom_unpaired=" << waiting_odom;
      if (!pending_factor_gnss_.empty()) {
        waiting_detail
            << " graph_first_stamp_ns="
            << stampNanoseconds(pending_factor_gnss_.front().stamp)
            << " graph_last_stamp_ns="
            << stampNanoseconds(pending_factor_gnss_.back().stamp);
      }
      queueTextEvent("GNSS_WAITING_AT_END_OF_STREAM", waiting_detail.str());
    }
    const std::int64_t conservation_delta = gnssConservationDelta();
    const std::uint64_t silent_drop_count = gnssSilentDropCount();
    if (conservation_delta != 0) {
      if (backend_error_.empty()) {
        backend_error_ = silent_drop_count != 0
                             ? "GNSS_SILENT_DROP_DETECTED"
                             : "GNSS_CONSERVATION_MISMATCH";
      }
      std::ostringstream error_detail;
      error_detail << "count=" << silent_drop_count
                   << " conservation_delta=" << conservation_delta;
      queueTextEvent("GNSS_CONSERVATION_ERROR", error_detail.str());
    }
    std::ostringstream detail;
    detail << "total_nodes=" << total_nodes_created_
           << " marginalized_nodes=" << marginalized_nodes_
           << " total_livo_factors=" << livo_factor_count_
           << " total_gnss_factors=" << gnss_factor_count_
           << " total_gnss_received=" << gnss_received_
           << " total_gnss_rejected=" << gnss_rejected_
           << " gnss_odom_only_rejected=" << gnss_odom_only_rejected_
           << " raw_received=" << raw_odom_received_
           << " raw_accepted=" << raw_odom_published_
           << " raw_duplicate=" << raw_odom_duplicate_
           << " raw_non_monotonic=" << raw_odom_non_monotonic_
           << " tf_published=" << tf_published_
           << " tf_duplicate_skipped=" << tf_duplicate_skipped_
           << " alignment_gnss_used=" << alignment_gnss_used_
           << " alignment_transition_to_graph_pending="
           << alignment_transition_to_graph_pending_
           << " alignment_transition_rejected="
           << alignment_transition_rejected_
           << " alignment_transition_waiting="
           << alignment_transition_waiting_
           << " gnss_duplicate_factor_count="
           << gnss_duplicate_factor_count_
           << " gnss_waiting_graph=" << waiting_graph
           << " gnss_waiting_alignment=" << waiting_alignment
           << " gnss_waiting_status=" << waiting_status
           << " gnss_waiting_odom_unpaired=" << waiting_odom
           << " gnss_silent_drop_count=" << silent_drop_count
           << " gnss_conservation_delta=" << conservation_delta;
    queueTextEvent("BACKEND_SUMMARY", detail.str());
    queueStatusCsv();
    batch = takePendingFileBatch();
  }
  writeFileBatch(batch, true);
  closeResultFiles();
}

void RtkFixedLagBackend::loadParameters(ros::NodeHandle &nh) {
  ros::NodeHandle params(nh, "rtk_backend");
  params.param("enable", config_.enable, config_.enable);
  params.param("raw_odom_topic", config_.raw_odom_topic,
               config_.raw_odom_topic);
  params.param("gnss_odom_topic", config_.gnss_odom_topic,
               config_.gnss_odom_topic);
  params.param("gnss_status_topic", config_.gnss_status_topic,
               config_.gnss_status_topic);
  params.param("optimized_odom_topic", config_.optimized_odom_topic,
               config_.optimized_odom_topic);
  params.param("optimized_path_topic", config_.optimized_path_topic,
               config_.optimized_path_topic);
  params.param("map_to_odom_topic", config_.map_to_odom_topic,
               config_.map_to_odom_topic);
  params.param("status_topic", config_.status_topic, config_.status_topic);
  params.param("lag_seconds", config_.lag_seconds, config_.lag_seconds);
  params.param("keyframe_translation_m", config_.keyframe_translation_m,
               config_.keyframe_translation_m);
  params.param("keyframe_rotation_deg", config_.keyframe_rotation_deg,
               config_.keyframe_rotation_deg);
  params.param("keyframe_max_interval_s", config_.keyframe_max_interval_s,
               config_.keyframe_max_interval_s);
  params.param("raw_odom_buffer_seconds", config_.raw_odom_buffer_seconds,
               config_.raw_odom_buffer_seconds);
  params.param("max_raw_odom_interpolation_gap_s",
               config_.max_raw_odom_interpolation_gap_s,
               config_.max_raw_odom_interpolation_gap_s);
  params.param("reuse_existing_node_time_diff_s",
               config_.reuse_existing_node_time_diff_s,
               config_.reuse_existing_node_time_diff_s);
  params.param("gnss_node_min_interval_s",
               config_.gnss_node_min_interval_s,
               config_.gnss_node_min_interval_s);
  params.param("max_active_states", config_.max_active_states,
               config_.max_active_states);
  params.param("alignment_min_pairs", config_.alignment_min_pairs,
               config_.alignment_min_pairs);
  params.param("alignment_min_baseline_m", config_.alignment_min_baseline_m,
               config_.alignment_min_baseline_m);
  params.param("alignment_max_pair_time_diff_s",
               config_.alignment_max_pair_time_diff_s,
               config_.alignment_max_pair_time_diff_s);
  params.param("alignment_max_rmse_m", config_.alignment_max_rmse_m,
               config_.alignment_max_rmse_m);
  params.param("min_gnss_sigma_xy_m", config_.min_gnss_sigma_xy_m,
               config_.min_gnss_sigma_xy_m);
  params.param("min_gnss_sigma_z_m", config_.min_gnss_sigma_z_m,
               config_.min_gnss_sigma_z_m);
  params.param("max_gnss_sigma_xy_m", config_.max_gnss_sigma_xy_m,
               config_.max_gnss_sigma_xy_m);
  params.param("max_gnss_sigma_z_m", config_.max_gnss_sigma_z_m,
               config_.max_gnss_sigma_z_m);
  params.param("max_gnss_residual_m", config_.max_gnss_residual_m,
               config_.max_gnss_residual_m);
  params.param("max_gnss_nis", config_.max_gnss_nis,
               config_.max_gnss_nis);
  params.param("robust_kernel", config_.robust_kernel,
               config_.robust_kernel);
  params.param("huber_delta", config_.huber_delta, config_.huber_delta);
  params.param("livo_translation_sigma_m",
               config_.livo_translation_sigma_m,
               config_.livo_translation_sigma_m);
  params.param("livo_rotation_sigma_rad", config_.livo_rotation_sigma_rad,
               config_.livo_rotation_sigma_rad);
  params.param("prior_translation_sigma_m",
               config_.prior_translation_sigma_m,
               config_.prior_translation_sigma_m);
  params.param("prior_roll_pitch_sigma_rad",
               config_.prior_roll_pitch_sigma_rad,
               config_.prior_roll_pitch_sigma_rad);
  params.param("prior_yaw_sigma_rad", config_.prior_yaw_sigma_rad,
               config_.prior_yaw_sigma_rad);
  params.param("frame_id", config_.frame_id, config_.frame_id);
  params.param("odom_frame_id", config_.odom_frame_id,
               config_.odom_frame_id);
  params.param("body_frame_id", config_.body_frame_id,
               config_.body_frame_id);
  params.param("log_interval_s", config_.log_interval_s,
               config_.log_interval_s);
  params.param("save_results", config_.save_results, config_.save_results);
  params.param("output_directory", config_.output_directory,
               config_.output_directory);
  params.param("raw_online_file", config_.raw_online_file,
               config_.raw_online_file);
  params.param("optimized_online_file", config_.optimized_online_file,
               config_.optimized_online_file);
  params.param("optimized_final_file", config_.optimized_final_file,
               config_.optimized_final_file);
  params.param("gnss_file", config_.gnss_file, config_.gnss_file);
  params.param("status_csv_file", config_.status_csv_file,
               config_.status_csv_file);
  params.param("flush_interval_s", config_.flush_interval_s,
               config_.flush_interval_s);
  params.param("save_text_log", config_.save_text_log,
               config_.save_text_log);
  params.param("text_log_file", config_.text_log_file,
               config_.text_log_file);

  std::vector<double> lever_arm{0.0, 0.0, 0.0};
  if (params.getParam("antenna_lever_arm_body_m", lever_arm)) {
    if (lever_arm.size() != 3) {
      throw std::invalid_argument(
          "rtk_backend/antenna_lever_arm_body_m must contain 3 values");
    }
    config_.antenna_lever_arm_body_m =
        gtsam::Point3(lever_arm[0], lever_arm[1], lever_arm[2]);
  }
}

void RtkFixedLagBackend::validateParameters() const {
  if (config_.lag_seconds <= 0.0 ||
      config_.keyframe_translation_m <= 0.0 ||
      config_.keyframe_rotation_deg <= 0.0 ||
      config_.keyframe_max_interval_s <= 0.0 ||
      config_.raw_odom_buffer_seconds <= 0.0 ||
      config_.max_raw_odom_interpolation_gap_s <= 0.0 ||
      config_.reuse_existing_node_time_diff_s < 0.0 ||
      config_.gnss_node_min_interval_s <= 0.0 ||
      config_.max_active_states <= 0 || config_.alignment_min_pairs < 2 ||
      config_.alignment_min_baseline_m <= 0.0 ||
      config_.alignment_max_pair_time_diff_s < 0.0 ||
      config_.alignment_max_rmse_m <= 0.0 ||
      config_.min_gnss_sigma_xy_m <= 0.0 ||
      config_.min_gnss_sigma_z_m <= 0.0 ||
      config_.max_gnss_sigma_xy_m < config_.min_gnss_sigma_xy_m ||
      config_.max_gnss_sigma_z_m < config_.min_gnss_sigma_z_m ||
      config_.max_gnss_residual_m <= 0.0 || config_.max_gnss_nis <= 0.0 ||
      config_.livo_translation_sigma_m <= 0.0 ||
      config_.livo_rotation_sigma_rad <= 0.0 ||
      config_.prior_translation_sigma_m <= 0.0 ||
      config_.prior_roll_pitch_sigma_rad <= 0.0 ||
      config_.prior_yaw_sigma_rad <= 0.0 || config_.log_interval_s <= 0.0 ||
      config_.flush_interval_s <= 0.0) {
    throw std::invalid_argument("invalid non-positive rtk_backend parameter");
  }
  if (config_.raw_odom_buffer_seconds <
      config_.max_raw_odom_interpolation_gap_s) {
    throw std::invalid_argument(
        "rtk_backend/raw_odom_buffer_seconds must be no smaller than the "
        "maximum interpolation gap");
  }
  if (config_.reuse_existing_node_time_diff_s >=
      config_.gnss_node_min_interval_s) {
    throw std::invalid_argument(
        "rtk_backend/reuse_existing_node_time_diff_s must be smaller than "
        "gnss_node_min_interval_s");
  }
  if (config_.robust_kernel != "huber" && config_.robust_kernel != "none") {
    throw std::invalid_argument(
        "rtk_backend/robust_kernel must be 'huber' or 'none'");
  }
  if (config_.robust_kernel == "huber" && config_.huber_delta <= 0.0) {
    throw std::invalid_argument("rtk_backend/huber_delta must be positive");
  }
  if ((config_.save_results || config_.save_text_log) &&
      config_.output_directory.empty()) {
    throw std::invalid_argument(
        "rtk_backend/output_directory must not be empty when saving output");
  }
}

void RtkFixedLagBackend::setupRos(ros::NodeHandle &nh) {
  raw_odom_subscriber_ = nh.subscribe(config_.raw_odom_topic, 500,
                                      &RtkFixedLagBackend::rawOdomCallback, this);
  gnss_odom_subscriber_ = nh.subscribe(config_.gnss_odom_topic, 200,
                                       &RtkFixedLagBackend::gnssOdomCallback,
                                       this);
  gnss_status_subscriber_ = nh.subscribe(
      config_.gnss_status_topic, 200,
      &RtkFixedLagBackend::gnssStatusCallback, this);
  optimized_odom_publisher_ =
      nh.advertise<nav_msgs::Odometry>(config_.optimized_odom_topic, 20);
  optimized_path_publisher_ =
      nh.advertise<nav_msgs::Path>(config_.optimized_path_topic, 2, true);
  map_to_odom_publisher_ = nh.advertise<geometry_msgs::TransformStamped>(
      config_.map_to_odom_topic, 20);
  status_publisher_ =
      nh.advertise<fast_livo::RtkBackendStatus>(config_.status_topic, 10, true);
  tf_broadcaster_.reset(new tf::TransformBroadcaster());
  status_timer_ = nh.createTimer(ros::Duration(config_.log_interval_s),
                                 &RtkFixedLagBackend::statusTimerCallback,
                                 this);
  flush_timer_ = nh.createTimer(ros::Duration(config_.flush_interval_s),
                                &RtkFixedLagBackend::flushTimerCallback,
                                this);
}

void RtkFixedLagBackend::initializeResultFiles() {
  if (!config_.save_results && !config_.save_text_log) return;
  namespace fs = std::filesystem;
  std::error_code error;
  fs::create_directories(config_.output_directory, error);
  if (error) {
    backend_error_ = "OUTPUT_DIRECTORY_CREATE_FAILED: " + error.message();
    ROS_ERROR_STREAM("[RTK_BACKEND_FILE] " << backend_error_
                     << " path=" << config_.output_directory);
    return;
  }

  const fs::path directory(config_.output_directory);
  if (config_.save_results) {
    raw_online_stream_.open(directory / config_.raw_online_file,
                            std::ios::out | std::ios::trunc);
    optimized_online_stream_.open(directory / config_.optimized_online_file,
                                  std::ios::out | std::ios::trunc);
    optimized_final_stream_.open(directory / config_.optimized_final_file,
                                 std::ios::out | std::ios::trunc);
    gnss_stream_.open(directory / config_.gnss_file,
                      std::ios::out | std::ios::trunc);
    status_csv_stream_.open(directory / config_.status_csv_file,
                            std::ios::out | std::ios::trunc);
  }
  if (config_.save_text_log) {
    text_log_stream_.open(directory / config_.text_log_file,
                          std::ios::out | std::ios::trunc);
  }

  const bool result_streams_ok =
      !config_.save_results ||
      (raw_online_stream_ && optimized_online_stream_ &&
       optimized_final_stream_ && gnss_stream_ && status_csv_stream_);
  const bool text_stream_ok = !config_.save_text_log || text_log_stream_;
  if (!result_streams_ok || !text_stream_ok) {
    backend_error_ = "RESULT_FILE_OPEN_FAILED";
    ROS_ERROR_STREAM("[RTK_BACKEND_FILE] " << backend_error_
                     << " directory=" << config_.output_directory
                     << "; backend will continue without file output");
    closeResultFiles();
    return;
  }

  result_files_ready_ = true;
  if (config_.save_results) {
    raw_online_stream_
        << "# frame_id=odom columns=timestamp tx ty tz qx qy qz qw\n";
    optimized_online_stream_
        << "# frame_id=map/ENU columns=timestamp tx ty tz qx qy qz qw; "
           "repeated timestamps may represent a later optimization of the "
           "current node\n";
    optimized_final_stream_
        << "# frame_id=map/ENU columns=timestamp tx ty tz qx qy qz qw; "
           "final fixed-lag timestamps are strictly increasing and unique\n";
    gnss_stream_
        << "# frame_id=map/ENU columns=timestamp x_m y_m z_m 0 0 0 1; "
           "unit quaternion is a placeholder because GNSS supplies no "
           "attitude\n";
    status_csv_stream_
        << "wall_time,ros_time,latest_sensor_stamp,initialized,"
           "alignment_ready,active_states,active_factors,active_livo_factors,"
           "active_gnss_factors,total_nodes,marginalized_nodes,"
           "total_livo_factors,total_gnss_received,total_gnss_factors,"
           "gnss_waiting,gnss_time_rejected,gnss_quality_rejected,"
           "gnss_too_old,gnss_interpolation_gap,gnss_interpolation_invalid,"
           "gnss_late,gnss_rate_limited,gnss_duplicate,gnss_no_active_state,"
           "raw_odom_received,raw_odom_published,raw_odom_duplicate,"
           "raw_odom_non_monotonic,tf_published,tf_duplicate_skipped,"
           "last_gnss_dt,last_gnss_residual,last_gnss_nis,optimization_ms,"
           "optimization_average_ms,optimization_max_ms,"
           "interpolation_average_gap_s,interpolation_max_gap_s,"
           "last_reject_reason,backend_error,total_gnss_rejected,"
           "gnss_odom_only_rejected,"
           "alignment_gnss_used,alignment_last_used_gnss_stamp_ns,"
           "alignment_transition_to_graph_pending,"
           "alignment_transition_rejected,alignment_transition_waiting,"
           "gnss_waiting_alignment,gnss_waiting_status,"
           "gnss_duplicate_factor_count,gnss_silent_drop_count,"
           "gnss_conservation_delta\n";
  }
}

void RtkFixedLagBackend::closeResultFiles() {
  std::lock_guard<std::mutex> lock(file_mutex_);
  auto close = [](std::ofstream &stream) {
    if (!stream.is_open()) return;
    stream.flush();
    stream.close();
  };
  close(raw_online_stream_);
  close(optimized_online_stream_);
  close(optimized_final_stream_);
  close(gnss_stream_);
  close(status_csv_stream_);
  close(text_log_stream_);
  result_files_ready_ = false;
}

AlignmentResult RtkFixedLagBackend::estimateSe2Alignment(
    const std::vector<AlignmentPair> &pairs) {
  AlignmentResult result;
  result.pair_count = pairs.size();
  if (pairs.size() < 2) return result;

  gtsam::Point3 odom_centroid(0.0, 0.0, 0.0);
  gtsam::Point3 enu_centroid(0.0, 0.0, 0.0);
  for (const AlignmentPair &pair : pairs) {
    odom_centroid += pair.odom_position;
    enu_centroid += pair.enu_position;
  }
  odom_centroid /= static_cast<double>(pairs.size());
  enu_centroid /= static_cast<double>(pairs.size());

  double cosine_term = 0.0;
  double sine_term = 0.0;
  double max_baseline_squared = 0.0;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const gtsam::Point3 odom_delta = pairs[i].odom_position - odom_centroid;
    const gtsam::Point3 enu_delta = pairs[i].enu_position - enu_centroid;
    cosine_term += odom_delta.x() * enu_delta.x() +
                   odom_delta.y() * enu_delta.y();
    sine_term += odom_delta.x() * enu_delta.y() -
                 odom_delta.y() * enu_delta.x();
    for (std::size_t j = 0; j < i; ++j) {
      const double dx = pairs[i].odom_position.x() -
                        pairs[j].odom_position.x();
      const double dy = pairs[i].odom_position.y() -
                        pairs[j].odom_position.y();
      max_baseline_squared =
          std::max(max_baseline_squared, dx * dx + dy * dy);
    }
  }
  result.baseline_m = std::sqrt(max_baseline_squared);
  if (std::hypot(cosine_term, sine_term) < 1e-9) return result;

  result.yaw_rad = std::atan2(sine_term, cosine_term);
  const double cosine = std::cos(result.yaw_rad);
  const double sine = std::sin(result.yaw_rad);
  result.translation = gtsam::Point3(
      enu_centroid.x() - cosine * odom_centroid.x() +
          sine * odom_centroid.y(),
      enu_centroid.y() - sine * odom_centroid.x() -
          cosine * odom_centroid.y(),
      enu_centroid.z() - odom_centroid.z());

  double squared_error_sum = 0.0;
  for (const AlignmentPair &pair : pairs) {
    const gtsam::Point3 predicted(
        cosine * pair.odom_position.x() - sine * pair.odom_position.y() +
            result.translation.x(),
        sine * pair.odom_position.x() + cosine * pair.odom_position.y() +
            result.translation.y(),
        pair.odom_position.z() + result.translation.z());
    squared_error_sum += (pair.enu_position - predicted).squaredNorm();
  }
  result.rmse_m =
      std::sqrt(squared_error_sum / static_cast<double>(pairs.size()));
  result.valid = true;
  return result;
}

bool RtkFixedLagBackend::shouldCreateKeyframe(
    const gtsam::Pose3 &previous, const gtsam::Pose3 &current,
    double interval_s, const BackendConfig &config) {
  const gtsam::Pose3 relative = previous.between(current);
  const double translation_m = relative.translation().norm();
  const double rotation_deg =
      gtsam::Rot3::Logmap(relative.rotation()).norm() * kRadToDeg;
  return translation_m >= config.keyframe_translation_m ||
         rotation_deg >= config.keyframe_rotation_deg ||
         interval_s >= config.keyframe_max_interval_s;
}

bool RtkFixedLagBackend::interpolatePose(
    const ros::Time &stamp0, const gtsam::Pose3 &pose0,
    const ros::Time &stamp1, const gtsam::Pose3 &pose1,
    const ros::Time &target, double max_gap_s, gtsam::Pose3 *interpolated,
    double *interval_s, std::string *reject_reason) {
  if (!interpolated || !interval_s || !reject_reason ||
      !poseIsFinite(pose0) || !poseIsFinite(pose1) ||
      target < stamp0 || target > stamp1) {
    if (reject_reason) *reject_reason = kInterpolationInvalid;
    return false;
  }
  const double gap_s = (stamp1 - stamp0).toSec();
  *interval_s = std::max(0.0, gap_s);
  if (target == stamp0) {
    *interpolated = pose0;
    return true;
  }
  if (target == stamp1) {
    *interpolated = pose1;
    return true;
  }
  if (!std::isfinite(gap_s) || gap_s <= 0.0) {
    *reject_reason = kInterpolationInvalid;
    return false;
  }
  if (gap_s > max_gap_s) {
    *reject_reason = kInterpolationGapTooLarge;
    return false;
  }

  const double alpha = (target - stamp0).toSec() / gap_s;
  if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
    *reject_reason = kInterpolationInvalid;
    return false;
  }
  const gtsam::Point3 translation =
      (1.0 - alpha) * pose0.translation() + alpha * pose1.translation();
  gtsam::Quaternion quaternion0 = pose0.rotation().toQuaternion();
  gtsam::Quaternion quaternion1 = pose1.rotation().toQuaternion();
  quaternion0.normalize();
  quaternion1.normalize();
  gtsam::Quaternion quaternion = quaternion0.slerp(alpha, quaternion1);
  if (!std::isfinite(quaternion.norm()) || quaternion.norm() < 1e-12) {
    *reject_reason = kInterpolationInvalid;
    return false;
  }
  quaternion.normalize();
  *interpolated = gtsam::Pose3(
      gtsam::Rot3::Quaternion(quaternion.w(), quaternion.x(), quaternion.y(),
                              quaternion.z()),
      translation);
  if (!poseIsFinite(*interpolated)) {
    *reject_reason = kInterpolationInvalid;
    return false;
  }
  reject_reason->clear();
  return true;
}

gtsam::Pose3 RtkFixedLagBackend::poseFromMessage(
    const geometry_msgs::Pose &pose) {
  const double norm = std::sqrt(
      pose.orientation.x * pose.orientation.x +
      pose.orientation.y * pose.orientation.y +
      pose.orientation.z * pose.orientation.z +
      pose.orientation.w * pose.orientation.w);
  if (!std::isfinite(norm) || norm < 1e-12 ||
      !std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) ||
      !std::isfinite(pose.position.z)) {
    throw std::invalid_argument("invalid odometry pose");
  }
  return gtsam::Pose3(
      gtsam::Rot3::Quaternion(pose.orientation.w / norm,
                              pose.orientation.x / norm,
                              pose.orientation.y / norm,
                              pose.orientation.z / norm),
      gtsam::Point3(pose.position.x, pose.position.y, pose.position.z));
}

geometry_msgs::Pose RtkFixedLagBackend::poseToMessage(
    const gtsam::Pose3 &pose) {
  geometry_msgs::Pose message;
  message.position.x = pose.x();
  message.position.y = pose.y();
  message.position.z = pose.z();
  const gtsam::Quaternion quaternion = pose.rotation().toQuaternion();
  message.orientation.w = quaternion.w();
  message.orientation.x = quaternion.x();
  message.orientation.y = quaternion.y();
  message.orientation.z = quaternion.z();
  return message;
}

bool RtkFixedLagBackend::poseIsFinite(const gtsam::Pose3 &pose) {
  return pose.matrix().allFinite();
}

void RtkFixedLagBackend::rawOdomCallback(
    const nav_msgs::OdometryConstPtr &message) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  ++raw_odom_received_;
  try {
    const std::int64_t stamp_ns = stampNanoseconds(message->header.stamp);
    if (last_raw_odom_stamp_ns_ >= 0 && stamp_ns <= last_raw_odom_stamp_ns_) {
      if (stamp_ns == last_raw_odom_stamp_ns_)
        ++raw_odom_duplicate_;
      else
        ++raw_odom_non_monotonic_;
      ROS_WARN_STREAM_THROTTLE(
          config_.log_interval_s,
          "[RTK_BACKEND_RAW_REJECT] duplicate=" << raw_odom_duplicate_
          << " non_monotonic=" << raw_odom_non_monotonic_
          << " received=" << raw_odom_received_
          << " accepted=" << raw_odom_published_);
      publishStatus();
      return;
    }

    RawOdomSample sample{message->header.stamp,
                         poseFromMessage(message->pose.pose)};
    if (!poseIsFinite(sample.pose)) {
      throw std::invalid_argument("raw odometry pose contains NaN/Inf");
    }
    last_raw_odom_stamp_ns_ = stamp_ns;
    ++raw_odom_published_;
    newest_sensor_stamp_ = std::max(newest_sensor_stamp_, sample.stamp);
    raw_odom_buffer_.push_back(sample);
    queueRawPose(sample);

    if (!alignment_.valid) tryCollectAlignmentPairs();
    if (alignment_.valid && !initialized_) initializeGraph(sample);
    if (initialized_ && !backend_halted_) {
      processPendingGnss();
      maybeAddKeyframe(sample);
    }
    pruneRawOdomBuffer();
    publishStatus();
  } catch (const std::exception &error) {
    backend_error_ = error.what();
    ROS_ERROR_STREAM_THROTTLE(config_.log_interval_s,
                              "[RTK_BACKEND_RAW_ERROR] " << error.what());
    publishStatus();
  }
}

void RtkFixedLagBackend::gnssStatusCallback(
    const fast_livo::GnssStatusConstPtr &message) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  newest_sensor_stamp_ = std::max(newest_sensor_stamp_, message->header.stamp);
  ++gnss_received_;
  const bool fixed = message->accepted &&
                     message->filtered_quality ==
                         fast_livo::GnssStatus::RTK_FIXED;
  if (!fixed) {
    if (!alignment_.valid) {
      resetAlignmentCollection(message->reject_reason.empty()
                                   ? "NOT_RTK_FIXED"
                                   : message->reject_reason);
    }
    ++gnss_quality_rejected_;
    rejectGnss(message->reject_reason.empty() ? "NOT_RTK_FIXED"
                                              : message->reject_reason,
               0.0, 0.0, &message->header.stamp);
    publishStatus();
    return;
  }

  const std::uint64_t stamp_ns = message->header.stamp.toNSec();
  if (pending_status_.count(stamp_ns) != 0) {
    rejectGnss(kDuplicateGnssTimestamp, 0.0, 0.0,
               &message->header.stamp);
    publishStatus();
    return;
  }
  pending_status_[stamp_ns] = *message;
  tryPairGnssMessages(stamp_ns);
  while (pending_status_.size() > kMaximumUnpairedGnssMessages) {
    rejectGnss("STATUS_ODOM_MISMATCH", 0.0, 0.0,
               &pending_status_.begin()->second.header.stamp);
    pending_status_.erase(pending_status_.begin());
  }
  publishStatus();
}

void RtkFixedLagBackend::gnssOdomCallback(
    const nav_msgs::OdometryConstPtr &message) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  newest_sensor_stamp_ = std::max(newest_sensor_stamp_, message->header.stamp);
  const std::uint64_t stamp_ns = message->header.stamp.toNSec();
  if (pending_gnss_odom_.count(stamp_ns) != 0) {
    ++gnss_odom_only_rejected_;
    rejectGnss(kDuplicateGnssTimestamp, 0.0, 0.0,
               &message->header.stamp);
    publishStatus();
    return;
  }
  pending_gnss_odom_[stamp_ns] = *message;
  tryPairGnssMessages(stamp_ns);
  while (pending_gnss_odom_.size() > kMaximumUnpairedGnssMessages)
    pending_gnss_odom_.erase(pending_gnss_odom_.begin());
  publishStatus();
}

void RtkFixedLagBackend::tryPairGnssMessages(std::uint64_t stamp_ns) {
  const auto status = pending_status_.find(stamp_ns);
  const auto odometry = pending_gnss_odom_.find(stamp_ns);
  if (status == pending_status_.end() ||
      odometry == pending_gnss_odom_.end()) {
    return;
  }
  processAcceptedGnss(odometry->second);
  pending_status_.erase(status);
  pending_gnss_odom_.erase(odometry);
}

void RtkFixedLagBackend::processAcceptedGnss(
    const nav_msgs::Odometry &odometry) {
  const std::int64_t stamp_ns = stampNanoseconds(odometry.header.stamp);
  if (last_enqueued_gnss_stamp_ns_ >= 0 &&
      stamp_ns <= last_enqueued_gnss_stamp_ns_) {
    rejectGnss(stamp_ns == last_enqueued_gnss_stamp_ns_
                   ? kDuplicateGnssTimestamp
                   : kGnssLateOutOfOrder,
               0.0, 0.0, &odometry.header.stamp);
    return;
  }

  const double covariance_x = odometry.pose.covariance[0];
  const double covariance_y = odometry.pose.covariance[7];
  const double covariance_z = odometry.pose.covariance[14];
  const geometry_msgs::Point &position = odometry.pose.pose.position;
  if (!std::isfinite(covariance_x) || !std::isfinite(covariance_y) ||
      !std::isfinite(covariance_z) || covariance_x <= 0.0 ||
      covariance_y <= 0.0 || covariance_z <= 0.0 ||
      !std::isfinite(position.x) || !std::isfinite(position.y) ||
      !std::isfinite(position.z)) {
    rejectGnss("INVALID_COVARIANCE_OR_POSITION", 0.0, 0.0,
               &odometry.header.stamp);
    return;
  }

  GnssMeasurement measurement;
  measurement.stamp = odometry.header.stamp;
  measurement.position = gtsam::Point3(position.x, position.y, position.z);
  measurement.sigmas = gtsam::Vector3(
      clamp(std::sqrt(covariance_x), config_.min_gnss_sigma_xy_m,
            config_.max_gnss_sigma_xy_m),
      clamp(std::sqrt(covariance_y), config_.min_gnss_sigma_xy_m,
            config_.max_gnss_sigma_xy_m),
      clamp(std::sqrt(covariance_z), config_.min_gnss_sigma_z_m,
            config_.max_gnss_sigma_z_m));
  last_enqueued_gnss_stamp_ns_ = stamp_ns;

  if (!alignment_.valid) {
    pending_alignment_gnss_.push_back(measurement);
    tryCollectAlignmentPairs();
  } else {
    insertPendingGnss(measurement);
    processPendingGnss();
  }
}

void RtkFixedLagBackend::insertPendingGnss(
    const GnssMeasurement &measurement) {
  const auto insertion = std::upper_bound(
      pending_factor_gnss_.begin(), pending_factor_gnss_.end(),
      measurement.stamp,
      [](const ros::Time &stamp, const GnssMeasurement &candidate) {
        return stamp < candidate.stamp;
      });
  pending_factor_gnss_.insert(insertion, measurement);
}

bool RtkFixedLagBackend::interpolateRawPose(
    const ros::Time &stamp, gtsam::Pose3 *pose, double *interval_s,
    std::string *reason) const {
  if (raw_odom_buffer_.empty()) {
    *reason = kWaitingForRawOdom;
    return false;
  }
  if (stamp < raw_odom_buffer_.front().stamp) {
    *reason = kGnssTooOldForBuffer;
    return false;
  }
  if (stamp > raw_odom_buffer_.back().stamp) {
    *reason = kWaitingForRawOdom;
    return false;
  }

  const auto upper = std::lower_bound(
      raw_odom_buffer_.begin(), raw_odom_buffer_.end(), stamp,
      [](const RawOdomSample &sample, const ros::Time &target) {
        return sample.stamp < target;
      });
  if (upper == raw_odom_buffer_.end()) {
    *reason = kWaitingForRawOdom;
    return false;
  }
  if (upper->stamp == stamp) {
    *pose = upper->pose;
    *interval_s = 0.0;
    reason->clear();
    return poseIsFinite(*pose);
  }
  if (upper == raw_odom_buffer_.begin()) {
    *reason = kGnssTooOldForBuffer;
    return false;
  }
  const auto lower = std::prev(upper);
  return interpolatePose(lower->stamp, lower->pose, upper->stamp, upper->pose,
                         stamp, config_.max_raw_odom_interpolation_gap_s,
                         pose, interval_s, reason);
}

void RtkFixedLagBackend::tryCollectAlignmentPairs() {
  if (raw_odom_buffer_.empty()) return;

  for (auto measurement = pending_alignment_gnss_.begin();
       measurement != pending_alignment_gnss_.end();) {
    if (measurement->stamp >= raw_odom_buffer_.front().stamp &&
        measurement->stamp <= raw_odom_buffer_.back().stamp) {
      const auto upper = std::lower_bound(
          raw_odom_buffer_.begin(), raw_odom_buffer_.end(), measurement->stamp,
          [](const RawOdomSample &sample, const ros::Time &stamp) {
            return sample.stamp < stamp;
          });
      double nearest_difference_s = std::numeric_limits<double>::infinity();
      if (upper != raw_odom_buffer_.end())
        nearest_difference_s =
            stampDifference(upper->stamp, measurement->stamp);
      if (upper != raw_odom_buffer_.begin())
        nearest_difference_s = std::min(
            nearest_difference_s,
            stampDifference(std::prev(upper)->stamp, measurement->stamp));
      if (nearest_difference_s >
          config_.alignment_max_pair_time_diff_s) {
        rejectGnss(kInterpolationGapTooLarge, 0.0, 0.0,
                   &measurement->stamp);
        last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement->stamp);
        measurement = pending_alignment_gnss_.erase(measurement);
        continue;
      }
    }
    gtsam::Pose3 raw_pose;
    double interval_s = 0.0;
    std::string reason;
    if (!interpolateRawPose(measurement->stamp, &raw_pose, &interval_s,
                            &reason)) {
      if (reason == kWaitingForRawOdom) break;
      rejectGnss(reason, 0.0, 0.0, &measurement->stamp);
      last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement->stamp);
      measurement = pending_alignment_gnss_.erase(measurement);
      continue;
    }

    const std::int64_t measurement_stamp_ns =
        stampNanoseconds(measurement->stamp);
    alignment_pairs_.push_back(AlignmentPair{
        raw_pose.transformFrom(config_.antenna_lever_arm_body_m),
        measurement->position, measurement_stamp_ns});
    ++alignment_gnss_used_;
    alignment_last_used_gnss_stamp_ns_ = measurement_stamp_ns;
    last_processed_gnss_stamp_ns_ = measurement_stamp_ns;
    measurement = pending_alignment_gnss_.erase(measurement);
    // ponytail: startup-only fixed storage cap; a future streaming alignment
    // estimator is the upgrade path for multi-minute stationary starts.
    if (alignment_pairs_.size() > kMaximumAlignmentPairs) {
      alignment_pairs_.erase(
          alignment_pairs_.begin(),
          alignment_pairs_.begin() + kAlignmentPairsToDropAtCapacity);
    }
  }

  if (tryFinishAlignment()) {
    if (!initialized_ && !raw_odom_buffer_.empty())
      initializeGraph(raw_odom_buffer_.back());
    if (initialized_ && !backend_halted_) processPendingGnss();
  }
}

void RtkFixedLagBackend::transitionPendingGnssAfterAlignment(
    std::int64_t alignment_cutoff_stamp_ns) {
  if (pending_alignment_gnss_.empty()) {
    std::ostringstream detail;
    detail << "last_used_alignment_gnss_stamp_ns="
           << alignment_cutoff_stamp_ns
           << " pending_before=0 used_for_alignment="
           << alignment_.pair_count
           << " moved_to_graph_pending=0 rejected=0 remaining_waiting=0";
    queueTextEvent("RTK_ALIGNMENT_TRANSITION", detail.str());
    ROS_INFO_STREAM("[RTK_ALIGNMENT_TRANSITION] " << detail.str());
    return;
  }

  std::stable_sort(
      pending_alignment_gnss_.begin(), pending_alignment_gnss_.end(),
      [](const GnssMeasurement &left, const GnssMeasurement &right) {
        return left.stamp < right.stamp;
      });
  const std::size_t pending_before = pending_alignment_gnss_.size();
  const std::int64_t oldest_raw_stamp_ns =
      stampNanoseconds(raw_odom_buffer_.front().stamp);
  const std::int64_t newest_raw_stamp_ns =
      stampNanoseconds(raw_odom_buffer_.back().stamp);
  std::size_t moved_to_graph_pending = 0;
  std::size_t rejected = 0;
  std::size_t remaining_waiting = 0;
  std::int64_t previous_stamp_ns = -1;
  while (!pending_alignment_gnss_.empty()) {
    GnssMeasurement measurement = pending_alignment_gnss_.front();
    pending_alignment_gnss_.pop_front();
    const std::int64_t measurement_stamp_ns =
        stampNanoseconds(measurement.stamp);
    if (measurement_stamp_ns != previous_stamp_ns &&
        measurement_stamp_ns > alignment_cutoff_stamp_ns &&
        measurement_stamp_ns >= oldest_raw_stamp_ns) {
      insertPendingGnss(measurement);
      ++moved_to_graph_pending;
      if (measurement_stamp_ns > newest_raw_stamp_ns) ++remaining_waiting;
      previous_stamp_ns = measurement_stamp_ns;
      continue;
    }

    ++alignment_transition_rejected_;
    ++rejected;
    const char *reason = measurement_stamp_ns == previous_stamp_ns
                             ? kDuplicateGnssTimestamp
                             : kAlignmentTransitionTooOld;
    rejectGnss(reason, 0.0, 0.0, &measurement.stamp);
    last_processed_gnss_stamp_ns_ = measurement_stamp_ns;
    previous_stamp_ns = measurement_stamp_ns;
  }

  alignment_transition_to_graph_pending_ += moved_to_graph_pending;
  alignment_transition_waiting_ += remaining_waiting;
  std::ostringstream detail;
  detail << "last_used_alignment_gnss_stamp_ns="
         << alignment_cutoff_stamp_ns
         << " pending_before=" << pending_before
         << " used_for_alignment=" << alignment_.pair_count
         << " moved_to_graph_pending=" << moved_to_graph_pending
         << " rejected=" << rejected
         << " remaining_waiting=" << remaining_waiting;
  queueTextEvent("RTK_ALIGNMENT_TRANSITION", detail.str());
  ROS_INFO_STREAM("[RTK_ALIGNMENT_TRANSITION] " << detail.str());
}

void RtkFixedLagBackend::resetAlignmentCollection(
    const std::string &reason) {
  if (!alignment_pairs_.empty() || !pending_alignment_gnss_.empty()) {
    ROS_WARN_STREAM_THROTTLE(
        config_.log_interval_s,
        "[RTK_BACKEND_ALIGNMENT_RESET] reason=" << reason
        << " pairs=" << alignment_pairs_.size());
  }
  for (const GnssMeasurement &measurement : pending_alignment_gnss_) {
    rejectGnss(kAlignmentReset, 0.0, 0.0, &measurement.stamp);
    last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement.stamp);
  }
  alignment_pairs_.clear();
  pending_alignment_gnss_.clear();
  alignment_last_used_gnss_stamp_ns_ = -1;
}

bool RtkFixedLagBackend::tryFinishAlignment() {
  if (alignment_.valid ||
      alignment_pairs_.size() <
          static_cast<std::size_t>(config_.alignment_min_pairs)) {
    return alignment_.valid;
  }
  const AlignmentResult candidate = estimateSe2Alignment(alignment_pairs_);
  alignment_ = candidate;
  if (!candidate.valid ||
      candidate.baseline_m < config_.alignment_min_baseline_m) {
    alignment_.valid = false;
    return false;
  }
  if (candidate.rmse_m > config_.alignment_max_rmse_m) {
    alignment_.valid = false;
    last_reject_reason_ = "ALIGNMENT_RMSE_TOO_LARGE";
    std::ostringstream detail;
    detail << "rmse=" << candidate.rmse_m
           << " max_rmse=" << config_.alignment_max_rmse_m
           << " pairs=" << candidate.pair_count
           << " baseline=" << candidate.baseline_m;
    queueTextEvent("ALIGNMENT_FAILED", detail.str());
    ROS_WARN_STREAM_THROTTLE(
        config_.log_interval_s,
        "[RTK_BACKEND_ALIGNMENT_REJECT] reason=ALIGNMENT_RMSE_TOO_LARGE "
            << detail.str());
    return false;
  }

  initial_map_to_odom_ = gtsam::Pose3(
      gtsam::Rot3::Rz(candidate.yaw_rad), candidate.translation);
  alignment_ = candidate;
  last_reject_reason_.clear();
  if (alignment_pairs_.empty() ||
      alignment_pairs_.back().gnss_stamp_ns < 0) {
    alignment_.valid = false;
    backend_error_ = "ALIGNMENT_CUTOFF_STAMP_MISSING";
    return false;
  }
  alignment_last_used_gnss_stamp_ns_ =
      alignment_pairs_.back().gnss_stamp_ns;
  transitionPendingGnssAfterAlignment(
      alignment_last_used_gnss_stamp_ns_);
  std::ostringstream detail;
  detail << "yaw_deg=" << candidate.yaw_rad * kRadToDeg
         << " translation=[" << candidate.translation.transpose() << "]"
         << " rmse=" << candidate.rmse_m
         << " pairs=" << candidate.pair_count
         << " baseline=" << candidate.baseline_m
         << " last_used_alignment_gnss_stamp_ns="
         << alignment_last_used_gnss_stamp_ns_;
  queueTextEvent("ALIGNMENT_SUCCESS", detail.str());
  ROS_INFO_STREAM("[RTK_BACKEND_ALIGNMENT] ready=1 " << detail.str());
  return true;
}

bool RtkFixedLagBackend::initializeGraph(const RawOdomSample &sample) {
  gtsam::ISAM2Params parameters;
  parameters.findUnusedFactorSlots = true;
  smoother_.reset(
      new gtsam::IncrementalFixedLagSmoother(config_.lag_seconds, parameters));

  const gtsam::Key key = gtsam::Symbol('x', next_keyframe_id_);
  const gtsam::Pose3 map_pose = initial_map_to_odom_.compose(sample.pose);
  gtsam::Vector6 sigmas;
  sigmas << config_.prior_roll_pitch_sigma_rad,
      config_.prior_roll_pitch_sigma_rad, config_.prior_yaw_sigma_rad,
      config_.prior_translation_sigma_m, config_.prior_translation_sigma_m,
      config_.prior_translation_sigma_m;
  const auto noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

  gtsam::NonlinearFactorGraph factors;
  factors.add(gtsam::PriorFactor<gtsam::Pose3>(key, map_pose, noise));
  gtsam::Values values;
  values.insert(key, map_pose);
  gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
  timestamps[key] = sample.stamp.toSec();
  if (!updateSmoother(factors, values, timestamps)) {
    smoother_.reset();
    return false;
  }

  keyframes_.push_back(Keyframe{next_keyframe_id_, key, sample.stamp,
                                sample.pose, map_pose, false});
  ++next_keyframe_id_;
  ++total_nodes_created_;
  initialized_ = true;
  queueTextEvent("NODE_CREATED", "id=0 type=prior");
  refreshEstimateAndPublish();
  ROS_INFO_STREAM("[RTK_BACKEND] initialized at sensor_stamp="
                  << sample.stamp.toSec());
  return true;
}

bool RtkFixedLagBackend::createGraphNode(const RawOdomSample &sample,
                                         bool gnss_triggered,
                                         gtsam::Key *created_key) {
  if (!initialized_ || backend_halted_ || keyframes_.empty()) return false;
  const Keyframe previous = keyframes_.back();
  if (sample.stamp <= previous.stamp) return false;

  const gtsam::Pose3 relative = previous.raw_pose.between(sample.pose);
  const gtsam::Pose3 initial = previous.optimized_pose.compose(relative);
  const gtsam::Key key = gtsam::Symbol('x', next_keyframe_id_);
  gtsam::Vector6 sigmas;
  // GTSAM Pose3 tangent order: rotation xyz followed by translation xyz.
  sigmas << config_.livo_rotation_sigma_rad,
      config_.livo_rotation_sigma_rad, config_.livo_rotation_sigma_rad,
      config_.livo_translation_sigma_m, config_.livo_translation_sigma_m,
      config_.livo_translation_sigma_m;
  const auto noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

  gtsam::NonlinearFactorGraph factors;
  factors.add(gtsam::BetweenFactor<gtsam::Pose3>(previous.key, key, relative,
                                                 noise));
  gtsam::Values values;
  values.insert(key, initial);
  gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
  timestamps[key] = sample.stamp.toSec();
  if (!updateSmoother(factors, values, timestamps)) return false;

  keyframes_.push_back(Keyframe{next_keyframe_id_, key, sample.stamp,
                                sample.pose, initial, gnss_triggered});
  ++next_keyframe_id_;
  ++total_nodes_created_;
  ++livo_factor_count_;
  if (created_key) *created_key = key;
  std::ostringstream detail;
  detail << "id=" << (next_keyframe_id_ - 1)
         << " stamp=" << std::setprecision(15) << sample.stamp.toSec()
         << " type=" << (gnss_triggered ? "gnss_triggered" : "motion");
  queueTextEvent("NODE_CREATED", detail.str());
  queueTextEvent("LIVO_FACTOR_ADDED", detail.str());
  refreshEstimateAndPublish(false);
  return true;
}

void RtkFixedLagBackend::maybeAddKeyframe(const RawOdomSample &sample) {
  if (!initialized_ || backend_halted_ || keyframes_.empty()) return;
  const Keyframe &previous = keyframes_.back();
  const double interval_s = (sample.stamp - previous.stamp).toSec();
  if (interval_s <= 0.0 ||
      !shouldCreateKeyframe(previous.raw_pose, sample.pose, interval_s,
                            config_)) {
    return;
  }
  if (createGraphNode(sample, false, nullptr)) refreshEstimateAndPublish();
}

RtkFixedLagBackend::Keyframe *RtkFixedLagBackend::findReusableKeyframe(
    const ros::Time &stamp, double *time_difference_s) {
  if (keyframes_.empty()) return nullptr;
  const auto nearest = std::min_element(
      keyframes_.begin(), keyframes_.end(),
      [&](const Keyframe &left, const Keyframe &right) {
        return stampDifference(left.stamp, stamp) <
               stampDifference(right.stamp, stamp);
      });
  *time_difference_s = stampDifference(nearest->stamp, stamp);
  return *time_difference_s <= config_.reuse_existing_node_time_diff_s
             ? &(*nearest)
             : nullptr;
}

RtkFixedLagBackend::Keyframe *RtkFixedLagBackend::findKeyframe(
    gtsam::Key key) {
  const auto found = std::find_if(
      keyframes_.begin(), keyframes_.end(),
      [&](const Keyframe &keyframe) { return keyframe.key == key; });
  return found == keyframes_.end() ? nullptr : &(*found);
}

void RtkFixedLagBackend::processPendingGnss() {
  if (!alignment_.valid || backend_halted_) return;

  for (auto measurement = pending_factor_gnss_.begin();
       measurement != pending_factor_gnss_.end();) {
    if (raw_odom_buffer_.empty() ||
        measurement->stamp > raw_odom_buffer_.back().stamp) {
      break;
    }
    if (measurement->stamp < raw_odom_buffer_.front().stamp) {
      rejectGnss(kGnssTooOldForBuffer, 0.0, 0.0,
                 &measurement->stamp);
      last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement->stamp);
      measurement = pending_factor_gnss_.erase(measurement);
      continue;
    }

    gtsam::Pose3 interpolated_raw_pose;
    double interpolation_gap_s = 0.0;
    std::string interpolation_reason;
    if (!interpolateRawPose(measurement->stamp, &interpolated_raw_pose,
                            &interpolation_gap_s, &interpolation_reason)) {
      if (interpolation_reason == kWaitingForRawOdom) break;
      rejectGnss(interpolation_reason, 0.0, 0.0,
                 &measurement->stamp);
      last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement->stamp);
      measurement = pending_factor_gnss_.erase(measurement);
      continue;
    }
    ++interpolation_count_;
    interpolation_gap_sum_s_ += interpolation_gap_s;
    interpolation_gap_max_s_ =
        std::max(interpolation_gap_max_s_, interpolation_gap_s);

    if (!initialized_ || keyframes_.empty() || !smoother_) {
      rejectGnss(kNoActiveGraphState, 0.0, 0.0,
                 &measurement->stamp);
      last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement->stamp);
      measurement = pending_factor_gnss_.erase(measurement);
      continue;
    }

    double association_dt_s = std::numeric_limits<double>::infinity();
    Keyframe *keyframe =
        findReusableKeyframe(measurement->stamp, &association_dt_s);
    bool node_created = false;
    if (!keyframe) {
      if (measurement->stamp <= keyframes_.back().stamp) {
        rejectGnss(kGnssLateOutOfOrder, 0.0, 0.0,
                   &measurement->stamp);
        last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement->stamp);
        measurement = pending_factor_gnss_.erase(measurement);
        continue;
      }
      const std::int64_t measurement_ns = stampNanoseconds(measurement->stamp);
      if (last_gnss_triggered_node_stamp_ns_ >= 0 &&
          static_cast<double>(measurement_ns -
                              last_gnss_triggered_node_stamp_ns_) *
                  1e-9 <
              config_.gnss_node_min_interval_s) {
        rejectGnss(kGnssRateLimited, 0.0, 0.0,
                   &measurement->stamp);
        last_processed_gnss_stamp_ns_ = measurement_ns;
        measurement = pending_factor_gnss_.erase(measurement);
        continue;
      }

      gtsam::Key created_key = 0;
      if (!createGraphNode(
              RawOdomSample{measurement->stamp, interpolated_raw_pose}, true,
              &created_key)) {
        rejectGnss(kNoActiveGraphState, 0.0, 0.0,
                   &measurement->stamp);
        last_processed_gnss_stamp_ns_ = measurement_ns;
        measurement = pending_factor_gnss_.erase(measurement);
        continue;
      }
      node_created = true;
      last_gnss_triggered_node_stamp_ns_ = measurement_ns;
      keyframe = findKeyframe(created_key);
      association_dt_s = 0.0;
      if (!keyframe) {
        rejectGnss(kNoActiveGraphState, 0.0, 0.0,
                   &measurement->stamp);
        refreshEstimateAndPublish();
        last_processed_gnss_stamp_ns_ = measurement_ns;
        measurement = pending_factor_gnss_.erase(measurement);
        continue;
      }
    }

    last_gnss_dt_s_ = association_dt_s;
    const bool factor_added = addGnssFactor(*measurement, *keyframe);
    if (factor_added || node_created) refreshEstimateAndPublish();
    last_processed_gnss_stamp_ns_ = stampNanoseconds(measurement->stamp);
    measurement = pending_factor_gnss_.erase(measurement);
  }

  const std::size_t maximum_pending =
      static_cast<std::size_t>(config_.max_active_states) * 10;
  while (pending_factor_gnss_.size() > maximum_pending) {
    rejectGnss(kGnssTooOldForBuffer, 0.0, 0.0,
               &pending_factor_gnss_.front().stamp);
    pending_factor_gnss_.pop_front();
  }
}

bool RtkFixedLagBackend::addGnssFactor(
    const GnssMeasurement &measurement, const Keyframe &keyframe) {
  const std::int64_t measurement_stamp_ns =
      stampNanoseconds(measurement.stamp);
  if (last_added_gnss_factor_stamp_ns_ >= 0 &&
      measurement_stamp_ns <= last_added_gnss_factor_stamp_ns_) {
    const bool duplicate =
        measurement_stamp_ns == last_added_gnss_factor_stamp_ns_;
    if (duplicate) ++gnss_duplicate_factor_count_;
    rejectGnss(duplicate ? kDuplicateGnssTimestamp : kGnssLateOutOfOrder,
               0.0, 0.0, &measurement.stamp);
    return false;
  }

  gtsam::Pose3 pose = keyframe.optimized_pose;
  try {
    pose = smoother_->calculateEstimate<gtsam::Pose3>(keyframe.key);
  } catch (const std::exception &) {
    rejectGnss(kNoActiveGraphState, 0.0, 0.0, &measurement.stamp);
    return false;
  }

  gtsam::Matrix36 antenna_jacobian;
  const gtsam::Point3 predicted = pose.transformFrom(
      config_.antenna_lever_arm_body_m, antenna_jacobian);
  const gtsam::Vector3 residual = measurement.position - predicted;
  const double residual_m = residual.norm();
  gtsam::Matrix3 innovation_covariance =
      antenna_jacobian * smoother_->marginalCovariance(keyframe.key) *
      antenna_jacobian.transpose();
  innovation_covariance.diagonal() +=
      measurement.sigmas.array().square().matrix();
  const Eigen::LDLT<gtsam::Matrix3> decomposition(innovation_covariance);
  if (decomposition.info() != Eigen::Success) {
    rejectGnss("INVALID_INNOVATION_COVARIANCE", residual_m, 0.0,
               &measurement.stamp);
    return false;
  }
  const double nis = residual.dot(decomposition.solve(residual));
  if (!std::isfinite(nis) || nis < 0.0) {
    rejectGnss("INVALID_INNOVATION_COVARIANCE", residual_m, nis,
               &measurement.stamp);
    return false;
  }
  last_gnss_residual_m_ = residual_m;
  last_gnss_nis_ = nis;
  if (residual_m > config_.max_gnss_residual_m) {
    rejectGnss("GNSS_RESIDUAL_TOO_LARGE", residual_m, nis,
               &measurement.stamp);
    return false;
  }
  if (nis > config_.max_gnss_nis) {
    rejectGnss("GNSS_NIS_TOO_LARGE", residual_m, nis,
               &measurement.stamp);
    return false;
  }

  gtsam::SharedNoiseModel noise =
      gtsam::noiseModel::Diagonal::Sigmas(measurement.sigmas);
  if (config_.robust_kernel == "huber") {
    noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(config_.huber_delta),
        noise);
  }
  gtsam::NonlinearFactorGraph factors;
  factors.add(boost::make_shared<GnssPositionArmFactor>(
      keyframe.key, measurement.position, config_.antenna_lever_arm_body_m,
      noise));
  if (!updateSmoother(factors, gtsam::Values(),
                      gtsam::FixedLagSmoother::KeyTimestampMap())) {
    rejectGnss("OPTIMIZATION_FAILED", residual_m, nis,
               &measurement.stamp);
    return false;
  }

  ++gnss_accepted_;
  ++gnss_factor_count_;
  last_added_gnss_factor_stamp_ns_ = measurement_stamp_ns;
  last_reject_reason_.clear();
  queueGnssPosition(measurement);
  std::ostringstream detail;
  detail << "key=" << keyframe.id
         << " stamp=" << std::setprecision(15) << measurement.stamp.toSec()
         << " dt=" << last_gnss_dt_s_ << " residual=" << residual_m
         << " nis=" << nis << " sigmas=[" << measurement.sigmas.transpose()
         << "]";
  queueTextEvent("GNSS_FACTOR_ADDED", detail.str());
  ROS_INFO_STREAM_THROTTLE(config_.log_interval_s,
                           "[RTK_BACKEND_GNSS_FACTOR] " << detail.str());
  return true;
}

std::vector<RtkFixedLagBackend::ArchiveRecord>
RtkFixedLagBackend::collectMarginalizationCandidates(
    const gtsam::FixedLagSmoother::KeyTimestampMap &timestamps) const {
  std::vector<ArchiveRecord> candidates;
  if (!initialized_ || !smoother_ || timestamps.empty() || keyframes_.empty())
    return candidates;

  double current_timestamp = -std::numeric_limits<double>::infinity();
  for (const auto &timestamp : smoother_->timestamps())
    current_timestamp = std::max(current_timestamp, timestamp.second);
  for (const auto &timestamp : timestamps)
    current_timestamp = std::max(current_timestamp, timestamp.second);
  const double cutoff = current_timestamp - config_.lag_seconds;
  const gtsam::Values estimate = smoother_->calculateEstimate();
  for (const Keyframe &keyframe : keyframes_) {
    if (keyframe.stamp.toSec() >= cutoff) break;
    if (finalized_keys_.count(keyframe.key) != 0 ||
        !estimate.exists(keyframe.key)) {
      continue;
    }
    candidates.push_back(ArchiveRecord{
        keyframe.key, keyframe.stamp,
        estimate.at<gtsam::Pose3>(keyframe.key)});
  }
  return candidates;
}

void RtkFixedLagBackend::commitMarginalizedArchives(
    const std::vector<ArchiveRecord> &candidates) {
  if (!smoother_) return;
  for (const ArchiveRecord &record : candidates) {
    if (smoother_->timestamps().count(record.key) != 0) continue;
    queueFinalPose(record);
    ++marginalized_nodes_;
    std::ostringstream detail;
    detail << "key=" << gtsam::DefaultKeyFormatter(record.key)
           << " stamp=" << std::setprecision(15) << record.stamp.toSec();
    queueTextEvent("STATE_MARGINALIZED", detail.str());
  }
}

bool RtkFixedLagBackend::updateSmoother(
    const gtsam::NonlinearFactorGraph &factors, const gtsam::Values &values,
    const gtsam::FixedLagSmoother::KeyTimestampMap &timestamps) {
  if (!smoother_ || backend_halted_) return false;
  const std::vector<ArchiveRecord> candidates =
      collectMarginalizationCandidates(timestamps);
  const auto start = std::chrono::steady_clock::now();
  try {
    smoother_->update(factors, values, timestamps);
    const auto end = std::chrono::steady_clock::now();
    optimization_time_ms_ =
        std::chrono::duration<double, std::milli>(end - start).count();
    ++optimization_count_;
    optimization_time_sum_ms_ += optimization_time_ms_;
    optimization_time_max_ms_ =
        std::max(optimization_time_max_ms_, optimization_time_ms_);
    commitMarginalizedArchives(candidates);
    backend_error_.clear();
    return true;
  } catch (const std::exception &error) {
    const auto end = std::chrono::steady_clock::now();
    optimization_time_ms_ =
        std::chrono::duration<double, std::milli>(end - start).count();
    backend_error_ = error.what();
    backend_halted_ = true;
    queueTextEvent("BACKEND_SUMMARY", "optimization_halted=" + backend_error_);
    ROS_ERROR_STREAM("[RTK_BACKEND_OPTIMIZATION_ERROR] backend halted: "
                     << error.what());
    return false;
  }
}

void RtkFixedLagBackend::archiveActiveStates() {
  if (!initialized_ || !smoother_) return;
  try {
    const gtsam::Values estimate = smoother_->calculateEstimate();
    for (const Keyframe &keyframe : keyframes_) {
      if (finalized_keys_.count(keyframe.key) != 0 ||
          !estimate.exists(keyframe.key)) {
        continue;
      }
      queueFinalPose(ArchiveRecord{keyframe.key, keyframe.stamp,
                                   estimate.at<gtsam::Pose3>(keyframe.key)});
    }
  } catch (const std::exception &error) {
    backend_error_ = std::string("FINAL_ARCHIVE_FAILED: ") + error.what();
    ROS_ERROR_STREAM("[RTK_BACKEND_FILE] " << backend_error_);
  }
}

void RtkFixedLagBackend::refreshEstimateAndPublish(bool publish_current) {
  if (!initialized_ || !smoother_) return;
  const gtsam::FixedLagSmoother::KeyTimestampMap &timestamps =
      smoother_->timestamps();
  keyframes_.erase(
      std::remove_if(keyframes_.begin(), keyframes_.end(),
                     [&](const Keyframe &keyframe) {
                       return timestamps.count(keyframe.key) == 0;
                     }),
      keyframes_.end());
  if (keyframes_.empty()) {
    backend_error_ = "NO_ACTIVE_KEYFRAMES";
    backend_halted_ = true;
    return;
  }

  const gtsam::Values estimate = smoother_->calculateEstimate();
  for (Keyframe &keyframe : keyframes_) {
    if (estimate.exists(keyframe.key))
      keyframe.optimized_pose = estimate.at<gtsam::Pose3>(keyframe.key);
  }

  active_factors_ = 0;
  active_livo_factors_ = 0;
  active_gnss_factors_ = 0;
  for (const auto &factor : smoother_->getFactors()) {
    if (!factor) continue;
    ++active_factors_;
    if (boost::dynamic_pointer_cast<
            gtsam::BetweenFactor<gtsam::Pose3>>(factor)) {
      ++active_livo_factors_;
    } else if (boost::dynamic_pointer_cast<GnssPositionArmFactor>(factor)) {
      ++active_gnss_factors_;
    }
  }
  max_active_states_observed_ =
      std::max(max_active_states_observed_, keyframes_.size());
  if (keyframes_.size() > static_cast<std::size_t>(config_.max_active_states)) {
    backend_error_ = "MAX_ACTIVE_STATES_EXCEEDED";
    backend_halted_ = true;
    ROS_ERROR_STREAM("[RTK_BACKEND] " << backend_error_
                     << " active=" << keyframes_.size()
                     << " limit=" << config_.max_active_states);
  }
  if (!publish_current) return;

  const Keyframe &current = keyframes_.back();
  nav_msgs::Odometry optimized;
  optimized.header.stamp = current.stamp;
  optimized.header.frame_id = config_.frame_id;
  optimized.child_frame_id = config_.body_frame_id;
  optimized.pose.pose = poseToMessage(current.optimized_pose);
  optimized_odom_publisher_.publish(optimized);
  queueOptimizedOnlinePose(current);

  nav_msgs::Path path;
  path.header = optimized.header;
  path.poses.reserve(keyframes_.size());
  for (const Keyframe &keyframe : keyframes_) {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = keyframe.stamp;
    pose.header.frame_id = config_.frame_id;
    pose.pose = poseToMessage(keyframe.optimized_pose);
    path.poses.push_back(pose);
  }
  optimized_path_publisher_.publish(path);

  const std::int64_t optimized_stamp_ns = stampNanoseconds(current.stamp);
  if (last_tf_stamp_ns_ >= 0 && optimized_stamp_ns <= last_tf_stamp_ns_) {
    ++tf_duplicate_skipped_;
  } else {
    const gtsam::Pose3 map_to_odom =
        current.optimized_pose.compose(current.raw_pose.inverse());
    const geometry_msgs::Pose map_to_odom_pose = poseToMessage(map_to_odom);
    geometry_msgs::TransformStamped transform;
    transform.header.stamp = current.stamp;
    transform.header.frame_id = config_.frame_id;
    transform.child_frame_id = config_.odom_frame_id;
    transform.transform.translation.x = map_to_odom_pose.position.x;
    transform.transform.translation.y = map_to_odom_pose.position.y;
    transform.transform.translation.z = map_to_odom_pose.position.z;
    transform.transform.rotation = map_to_odom_pose.orientation;
    map_to_odom_publisher_.publish(transform);

    tf::Transform tf_transform;
    tf_transform.setOrigin(tf::Vector3(map_to_odom.x(), map_to_odom.y(),
                                       map_to_odom.z()));
    const gtsam::Quaternion quaternion =
        map_to_odom.rotation().toQuaternion();
    tf_transform.setRotation(tf::Quaternion(
        quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()));
    tf_broadcaster_->sendTransform(tf::StampedTransform(
        tf_transform, current.stamp, config_.frame_id,
        config_.odom_frame_id));
    last_tf_stamp_ns_ = optimized_stamp_ns;
    ++tf_published_;
  }
  publishStatus();
}

void RtkFixedLagBackend::pruneRawOdomBuffer() {
  if (raw_odom_buffer_.empty()) return;
  const ros::Time newest = raw_odom_buffer_.back().stamp;
  while (raw_odom_buffer_.size() > 2 &&
         (newest - raw_odom_buffer_.front().stamp).toSec() >
             config_.raw_odom_buffer_seconds) {
    raw_odom_buffer_.pop_front();
  }
}

void RtkFixedLagBackend::rejectGnss(const std::string &reason,
                                    double residual_m, double nis,
                                    const ros::Time *measurement_stamp) {
  ++gnss_rejected_;
  if (reason == kGnssTooOldForBuffer) {
    ++gnss_too_old_;
    ++gnss_time_rejected_;
  } else if (reason == kInterpolationGapTooLarge) {
    ++gnss_interpolation_gap_;
    ++gnss_time_rejected_;
  } else if (reason == kInterpolationInvalid) {
    ++gnss_interpolation_invalid_;
    ++gnss_time_rejected_;
  } else if (reason == kGnssLateOutOfOrder) {
    ++gnss_late_out_of_order_;
    ++gnss_time_rejected_;
  } else if (reason == kGnssRateLimited) {
    ++gnss_rate_limited_;
    ++gnss_time_rejected_;
  } else if (reason == kDuplicateGnssTimestamp) {
    ++gnss_duplicate_timestamp_;
    ++gnss_time_rejected_;
  } else if (reason == kNoActiveGraphState) {
    ++gnss_no_active_state_;
  } else if (reason == kAlignmentTransitionTooOld) {
    ++gnss_time_rejected_;
  }
  last_reject_reason_ = reason;
  if (reason == "GNSS_RESIDUAL_TOO_LARGE" ||
      reason == "GNSS_NIS_TOO_LARGE" ||
      reason == "INVALID_INNOVATION_COVARIANCE" ||
      reason == "OPTIMIZATION_FAILED") {
    last_gnss_residual_m_ = residual_m;
    last_gnss_nis_ = nis;
  }
  const ros::Time rejected_stamp =
      measurement_stamp ? *measurement_stamp : newest_sensor_stamp_;
  std::ostringstream detail;
  detail << "reason=" << reason << " stamp=" << std::setprecision(15)
         << rejected_stamp.toSec() << " stamp_ns="
         << stampNanoseconds(rejected_stamp) << " residual=" << residual_m
         << " nis=" << nis;
  queueTextEvent("GNSS_REJECTED", detail.str());
  ROS_WARN_STREAM_THROTTLE(config_.log_interval_s,
                           "[RTK_BACKEND_GNSS_REJECT] " << detail.str()
                           << " total=" << gnss_rejected_);
}

std::int64_t RtkFixedLagBackend::gnssConservationDelta() const {
  const std::uint64_t accounted =
      (gnss_rejected_ - gnss_odom_only_rejected_) +
      alignment_gnss_used_ + gnss_factor_count_ +
      static_cast<std::uint64_t>(pending_status_.size()) +
      static_cast<std::uint64_t>(pending_alignment_gnss_.size()) +
      static_cast<std::uint64_t>(pending_factor_gnss_.size());
  if (gnss_received_ >= accounted)
    return static_cast<std::int64_t>(gnss_received_ - accounted);
  return -static_cast<std::int64_t>(accounted - gnss_received_);
}

std::uint64_t RtkFixedLagBackend::gnssSilentDropCount() const {
  const std::int64_t delta = gnssConservationDelta();
  return delta > 0 ? static_cast<std::uint64_t>(delta) : 0;
}

void RtkFixedLagBackend::statusTimerCallback(const ros::TimerEvent &) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  publishStatus();
  queueStatusCsv();
  ROS_INFO_STREAM_THROTTLE(
      config_.log_interval_s,
      "[RTK_BACKEND_STATUS] initialized=" << initialized_
          << " alignment_ready=" << alignment_.valid
          << " active_states=" << keyframes_.size()
          << " active_factors=" << active_factors_
          << " active_livo=" << active_livo_factors_
          << " active_gnss=" << active_gnss_factors_
          << " total_nodes=" << total_nodes_created_
          << " gnss_waiting=" << pending_factor_gnss_.size()
          << " gnss_factors=" << gnss_factor_count_
          << " alignment_gnss_used=" << alignment_gnss_used_
          << " alignment_transition_to_graph_pending="
          << alignment_transition_to_graph_pending_
          << " alignment_transition_rejected="
          << alignment_transition_rejected_
          << " alignment_transition_waiting="
          << alignment_transition_waiting_
          << " gnss_silent_drop_count=" << gnssSilentDropCount()
          << " gnss_conservation_delta=" << gnssConservationDelta()
          << " livo_factors=" << livo_factor_count_
          << " raw_duplicate=" << raw_odom_duplicate_
          << " raw_non_monotonic=" << raw_odom_non_monotonic_
          << " tf_duplicate_skipped=" << tf_duplicate_skipped_
          << " optimization_ms=" << optimization_time_ms_
          << " last_reject=" << last_reject_reason_
          << " error=" << backend_error_);
}

void RtkFixedLagBackend::flushTimerCallback(const ros::TimerEvent &) {
  FileBatch batch;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    batch = takePendingFileBatch();
  }
  writeFileBatch(batch, true);
}

void RtkFixedLagBackend::publishStatus() {
  if (!config_.enable) return;
  fast_livo::RtkBackendStatus status;
  status.header.stamp = newest_sensor_stamp_;
  status.header.frame_id = config_.frame_id;
  status.initialized = initialized_;
  status.alignment_ready = alignment_.valid;
  status.active_keyframes = static_cast<std::uint32_t>(keyframes_.size());
  status.active_factors = static_cast<std::uint32_t>(active_factors_);
  status.active_livo_factors =
      static_cast<std::uint32_t>(active_livo_factors_);
  status.active_gnss_factors =
      static_cast<std::uint32_t>(active_gnss_factors_);
  status.max_active_states_observed =
      static_cast<std::uint32_t>(max_active_states_observed_);
  status.total_nodes_created = total_nodes_created_;
  status.marginalized_nodes = marginalized_nodes_;
  status.raw_odom_received = raw_odom_received_;
  status.raw_odom_published = raw_odom_published_;
  status.raw_odom_duplicate = raw_odom_duplicate_;
  status.raw_odom_non_monotonic = raw_odom_non_monotonic_;
  if (last_raw_odom_stamp_ns_ >= 0)
    status.last_raw_odom_stamp.fromNSec(last_raw_odom_stamp_ns_);
  status.tf_published = tf_published_;
  status.tf_duplicate_skipped = tf_duplicate_skipped_;
  if (last_tf_stamp_ns_ >= 0) status.last_tf_stamp.fromNSec(last_tf_stamp_ns_);
  status.gnss_received = gnss_received_;
  status.gnss_accepted = gnss_accepted_;
  status.gnss_rejected = gnss_rejected_;
  status.gnss_factors = gnss_factor_count_;
  status.livo_factors = livo_factor_count_;
  status.gnss_waiting = pending_factor_gnss_.size();
  status.gnss_time_rejected = gnss_time_rejected_;
  status.gnss_quality_rejected = gnss_quality_rejected_;
  status.gnss_too_old_for_buffer = gnss_too_old_;
  status.gnss_interpolation_gap_too_large = gnss_interpolation_gap_;
  status.gnss_interpolation_invalid = gnss_interpolation_invalid_;
  status.gnss_late_out_of_order = gnss_late_out_of_order_;
  status.gnss_rate_limited = gnss_rate_limited_;
  status.gnss_duplicate_timestamp = gnss_duplicate_timestamp_;
  status.gnss_no_active_graph_state = gnss_no_active_state_;
  status.last_gnss_dt = last_gnss_dt_s_;
  status.last_gnss_residual = last_gnss_residual_m_;
  status.last_gnss_nis = last_gnss_nis_;
  status.optimization_time_ms = optimization_time_ms_;
  status.optimization_average_ms =
      optimization_count_ == 0
          ? 0.0
          : optimization_time_sum_ms_ /
                static_cast<double>(optimization_count_);
  status.optimization_max_ms = optimization_time_max_ms_;
  status.interpolation_average_gap_s =
      interpolation_count_ == 0
          ? 0.0
          : interpolation_gap_sum_s_ /
                static_cast<double>(interpolation_count_);
  status.interpolation_max_gap_s = interpolation_gap_max_s_;
  status.oldest_state_age_s =
      keyframes_.empty() || newest_sensor_stamp_ < keyframes_.front().stamp
          ? 0.0
          : (newest_sensor_stamp_ - keyframes_.front().stamp).toSec();
  status.lag_seconds = config_.lag_seconds;
  status.oldest_timestamp =
      keyframes_.empty() ? ros::Time() : keyframes_.front().stamp;
  status.newest_timestamp =
      keyframes_.empty() ? ros::Time() : keyframes_.back().stamp;
  status.alignment_yaw_deg = alignment_.yaw_rad * kRadToDeg;
  status.alignment_translation.x = alignment_.translation.x();
  status.alignment_translation.y = alignment_.translation.y();
  status.alignment_translation.z = alignment_.translation.z();
  status.alignment_rmse = alignment_.rmse_m;
  status.alignment_pair_count =
      static_cast<std::uint32_t>(alignment_.pair_count);
  status.alignment_baseline = alignment_.baseline_m;
  status.last_reject_reason = last_reject_reason_;
  status.backend_error = backend_error_;
  status_publisher_.publish(status);
}

std::string RtkFixedLagBackend::tumLine(const ros::Time &stamp,
                                        const gtsam::Pose3 &pose) {
  const gtsam::Quaternion quaternion = pose.rotation().toQuaternion();
  std::ostringstream line;
  line << std::fixed << std::setprecision(9) << stamp.toSec() << " "
       << std::setprecision(12) << pose.x() << " " << pose.y() << " "
       << pose.z() << " " << quaternion.x() << " " << quaternion.y() << " "
       << quaternion.z() << " " << quaternion.w();
  return line.str();
}

std::string RtkFixedLagBackend::gnssTumLine(
    const GnssMeasurement &measurement) {
  std::ostringstream line;
  line << std::fixed << std::setprecision(9) << measurement.stamp.toSec()
       << " " << std::setprecision(12) << measurement.position.x() << " "
       << measurement.position.y() << " " << measurement.position.z()
       << " 0 0 0 1";
  return line.str();
}

void RtkFixedLagBackend::queueRawPose(const RawOdomSample &sample) {
  if (config_.save_results) pending_file_batch_.raw_lines.push_back(
      tumLine(sample.stamp, sample.pose));
}

void RtkFixedLagBackend::queueOptimizedOnlinePose(
    const Keyframe &keyframe) {
  if (config_.save_results)
    pending_file_batch_.optimized_online_lines.push_back(
        tumLine(keyframe.stamp, keyframe.optimized_pose));
}

void RtkFixedLagBackend::queueFinalPose(const ArchiveRecord &record) {
  if (!finalized_keys_.insert(record.key).second) return;
  if (config_.save_results)
    pending_file_batch_.optimized_final_lines.push_back(
        tumLine(record.stamp, record.pose));
}

void RtkFixedLagBackend::queueGnssPosition(
    const GnssMeasurement &measurement) {
  if (config_.save_results)
    pending_file_batch_.gnss_lines.push_back(gnssTumLine(measurement));
}

void RtkFixedLagBackend::queueStatusCsv() {
  if (!config_.save_results) return;
  const double optimization_average =
      optimization_count_ == 0
          ? 0.0
          : optimization_time_sum_ms_ /
                static_cast<double>(optimization_count_);
  const double interpolation_average =
      interpolation_count_ == 0
          ? 0.0
          : interpolation_gap_sum_s_ /
                static_cast<double>(interpolation_count_);
  const std::int64_t conservation_delta = gnssConservationDelta();
  std::ostringstream line;
  line << std::fixed << std::setprecision(9) << ros::WallTime::now().toSec()
       << "," << ros::Time::now().toSec() << ","
       << newest_sensor_stamp_.toSec() << "," << initialized_ << ","
       << alignment_.valid << "," << keyframes_.size() << ","
       << active_factors_ << "," << active_livo_factors_ << ","
       << active_gnss_factors_ << "," << total_nodes_created_ << ","
       << marginalized_nodes_ << "," << livo_factor_count_ << ","
       << gnss_received_ << "," << gnss_factor_count_ << ","
       << pending_factor_gnss_.size() << "," << gnss_time_rejected_ << ","
       << gnss_quality_rejected_ << "," << gnss_too_old_ << ","
       << gnss_interpolation_gap_ << "," << gnss_interpolation_invalid_ << ","
       << gnss_late_out_of_order_ << "," << gnss_rate_limited_ << ","
       << gnss_duplicate_timestamp_ << "," << gnss_no_active_state_ << ","
       << raw_odom_received_ << "," << raw_odom_published_ << ","
       << raw_odom_duplicate_ << "," << raw_odom_non_monotonic_ << ","
       << tf_published_ << "," << tf_duplicate_skipped_ << ","
       << last_gnss_dt_s_ << "," << last_gnss_residual_m_ << ","
       << last_gnss_nis_ << "," << optimization_time_ms_ << ","
       << optimization_average << "," << optimization_time_max_ms_ << ","
       << interpolation_average << "," << interpolation_gap_max_s_ << ","
       << csvField(last_reject_reason_) << "," << csvField(backend_error_)
       << "," << gnss_rejected_ << "," << gnss_odom_only_rejected_ << ","
       << alignment_gnss_used_ << ","
       << alignment_last_used_gnss_stamp_ns_ << ","
       << alignment_transition_to_graph_pending_ << ","
       << alignment_transition_rejected_ << ","
       << alignment_transition_waiting_ << ","
       << pending_alignment_gnss_.size() << "," << pending_status_.size()
       << "," << gnss_duplicate_factor_count_ << ","
       << gnssSilentDropCount() << "," << conservation_delta;
  pending_file_batch_.status_lines.push_back(line.str());
}

void RtkFixedLagBackend::queueTextEvent(const std::string &event,
                                        const std::string &detail) {
  if (!config_.save_text_log) return;
  std::ostringstream line;
  line << std::fixed << std::setprecision(9) << newest_sensor_stamp_.toSec()
       << " " << event;
  if (!detail.empty()) line << " " << detail;
  pending_file_batch_.text_lines.push_back(line.str());
}

RtkFixedLagBackend::FileBatch RtkFixedLagBackend::takePendingFileBatch() {
  FileBatch batch;
  std::swap(batch, pending_file_batch_);
  return batch;
}

void RtkFixedLagBackend::writeFileBatch(const FileBatch &batch, bool flush) {
  if (!result_files_ready_) return;
  std::lock_guard<std::mutex> lock(file_mutex_);
  const auto write_lines = [](std::ofstream &stream,
                              const std::vector<std::string> &lines) {
    if (!stream.is_open()) return;
    for (const std::string &line : lines) stream << line << '\n';
  };
  write_lines(raw_online_stream_, batch.raw_lines);
  write_lines(optimized_online_stream_, batch.optimized_online_lines);
  write_lines(optimized_final_stream_, batch.optimized_final_lines);
  write_lines(gnss_stream_, batch.gnss_lines);
  write_lines(status_csv_stream_, batch.status_lines);
  write_lines(text_log_stream_, batch.text_lines);
  if (!flush) return;
  if (raw_online_stream_) raw_online_stream_.flush();
  if (optimized_online_stream_) optimized_online_stream_.flush();
  if (optimized_final_stream_) optimized_final_stream_.flush();
  if (gnss_stream_) gnss_stream_.flush();
  if (status_csv_stream_) status_csv_stream_.flush();
  if (text_log_stream_) text_log_stream_.flush();
}

}  // namespace fast_livo_backend
