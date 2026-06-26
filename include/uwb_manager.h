/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#ifndef UWB_MANAGER_H
#define UWB_MANAGER_H

#include "common_lib.h"

#include <atomic>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct UwbRangeMeasurement
{
  int anchor_id = -1;
  double range_m = 0.0;
  double stamp = 0.0;
  std::string raw_line;
};

struct UwbAnchor
{
  int id = -1;
  bool enabled = false;
  bool estimated = false;
  V3D position_w = V3D::Zero();
};

struct UwbAnchorSample
{
  V3D tag_position_w = V3D::Zero();
  double range_m = 0.0;
  double stamp = 0.0;
};

struct UwbAnchorDistanceConstraint
{
  int id_a = -1;
  int id_b = -1;
  double distance_m = 0.0;
};

class UwbManager
{
public:
  UwbManager();
  ~UwbManager();

  bool initialize(ros::NodeHandle &nh, const std::string &save_path);
  void shutdown();
  bool isRunning() const { return running_.load(); }
  bool updateEnabled() const { return update_en_; }
  int applyRangeUpdate(StatesGroup &state);

private:
  bool loadParameters(ros::NodeHandle &nh);
  bool openSerial();
  bool configureSerial();
  void closeSerial();
  void readLoop();
  bool loadReplayFile();
  std::vector<UwbRangeMeasurement> takeReplayMeasurements(double now);
  void handleLine(const std::string &line, double stamp);
  std::vector<UwbRangeMeasurement> parseLine(const std::string &line, double stamp) const;
  std::vector<UwbRangeMeasurement> takeRecentMeasurements(double now);
  void logRawLine(double stamp, const std::string &line, const std::vector<UwbRangeMeasurement> &measurements);
  void logEvent(double stamp, const std::string &level, const std::string &message);
  void logEventThrottled(double stamp, const std::string &key, double period_s,
                         const std::string &level, const std::string &message);
  void logUpdate(double stamp, int used_count, double residual_norm, const V3D &rot_add,
                 const V3D &trans_add, const V3D &tag_offset_add,
                 const std::string &range_details);
  int applyLatestMeasurements(StatesGroup &state, const std::vector<UwbRangeMeasurement> &measurements);
  bool tryAlignAnchorFrame(const StatesGroup &state,
                           const std::vector<UwbRangeMeasurement> &measurements);
  void collectAnchorFrameAlignSamples(const StatesGroup &state,
                                      const std::vector<UwbRangeMeasurement> &measurements);
  bool estimateMultiAnchorFrame(M3D &R_ext_to_w, V3D &t_ext_to_w,
                                double &rmse, int &used_anchor_count,
                                int &used_sample_count) const;
  bool tryInitializeBaselineAnchors(const StatesGroup &state,
                                    const std::vector<UwbRangeMeasurement> &measurements);
  double configuredBaselineDistance() const;
  void collectAnchorEstimateSamples(const StatesGroup &state, const std::vector<UwbRangeMeasurement> &measurements);
  bool estimateAnchorPosition(int anchor_id, UwbAnchor &anchor, double &rmse, int &rank,
                              std::string *failure_reason = nullptr) const;
  void applyAnchorDistanceConstraints();
  void logAnchorEstimate(int anchor_id, const V3D &position_w, double rmse, int rank, int sample_count);

  bool en_ = false;
  bool update_en_ = true;
  std::string input_source_ = "serial";
  std::string serial_port_ = "/dev/ttyUSB0";
  int baudrate_ = 115200;
  bool dtr_high_ = true;
  bool rts_high_ = false;
  std::string mode_ = "external_anchors";
  std::string parser_mode_ = "uwb";
  std::string log_filename_ = "uwb_ranges.txt";
  int log_flush_stride_ = 1;
  std::string replay_file_;
  std::string replay_time_mode_ = "relative";
  double replay_speed_ = 1.0;
  double range_scale_ = 1.0;
  double min_range_m_ = 0.05;
  double max_range_m_ = 250.0;
  double max_age_s_ = 0.5;
  int max_queue_size_ = 512;
  int min_update_anchors_ = 1;
  double range_noise_m_ = 0.20;
  double position_cov_floor_m_ = 0.0;
  double max_residual_m_ = 3.0;
  double update_max_rot_step_deg_ = 1.0;
  double update_max_trans_step_m_ = 0.10;
  V3D tag_offset_body_ = V3D::Zero();
  bool tag_offset_estimate_en_ = false;
  int tag_offset_estimate_min_anchors_ = 2;
  double tag_offset_init_cov_m_ = 0.10;
  double tag_offset_process_noise_m_ = 0.0;
  double tag_offset_update_max_step_m_ = 0.01;
  double tag_offset_max_norm_m_ = 1.0;
  V3D tag_offset_est_body_ = V3D::Zero();
  M3D tag_offset_cov_ = M3D::Identity() * 0.01;
  bool anchor_position_estimate_en_ = false;
  bool anchor_estimate_use_for_update_ = true;
  bool anchor_estimate_freeze_after_init_ = true;
  int anchor_estimate_min_samples_ = 30;
  int anchor_estimate_max_samples_ = 300;
  int anchor_estimate_min_rank_ = 2;
  double anchor_estimate_min_motion_m_ = 1.0;
  double anchor_estimate_max_rmse_m_ = 0.50;
  double anchor_estimate_max_step_m_ = 2.0;
  bool baseline_anchor_init_en_ = false;
  int baseline_anchor_start_id_ = 0;
  int baseline_anchor_end_id_ = 1;
  double baseline_distance_m_ = 0.0;
  double baseline_init_min_motion_m_ = 20.0;
  bool baseline_use_start_range_offset_ = true;
  bool baseline_anchors_initialized_ = false;
  bool baseline_start_pose_ready_ = false;
  bool baseline_start_range_ready_ = false;
  V3D baseline_start_tag_position_w_ = V3D::Zero();
  double baseline_start_range_m_ = 0.0;
  bool start_anchor_origin_en_ = false;
  int start_anchor_origin_id_ = 0;
  double start_anchor_origin_tolerance_m_ = 0.20;
  bool anchor_frame_align_en_ = false;
  int anchor_frame_align_start_id_ = 0;
  int anchor_frame_align_end_id_ = 1;
  double anchor_frame_align_min_motion_m_ = 20.0;
  bool anchor_frame_align_use_start_range_offset_ = true;
  bool anchor_frame_align_yaw_only_ = true;
  bool anchor_frame_align_multi_en_ = true;
  int anchor_frame_align_multi_min_anchors_ = 3;
  int anchor_frame_align_multi_min_samples_per_anchor_ = 5;
  int anchor_frame_align_multi_min_total_samples_ = 30;
  int anchor_frame_align_multi_max_samples_per_anchor_ = 200;
  int anchor_frame_align_multi_max_iterations_ = 15;
  double anchor_frame_align_multi_huber_delta_m_ = 1.0;
  double anchor_frame_align_multi_max_rmse_m_ = 3.0;
  double anchor_frame_align_multi_retry_period_s_ = 1.0;
  double anchor_frame_align_last_multi_attempt_stamp_ = -1.0;
  bool anchor_frame_aligned_ = false;
  bool anchor_frame_align_start_pose_ready_ = false;
  bool anchor_frame_align_start_range_ready_ = false;
  V3D anchor_frame_align_start_tag_position_w_ = V3D::Zero();
  double anchor_frame_align_start_range_m_ = 0.0;
  M3D anchor_frame_align_R_ext_to_w_ = M3D::Identity();
  V3D anchor_frame_align_t_ext_to_w_ = V3D::Zero();
  std::map<int, std::deque<UwbAnchorSample>> anchor_frame_align_samples_;

  int serial_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread read_thread_;
  mutable std::mutex measurement_mutex_;
  std::deque<UwbRangeMeasurement> measurement_queue_;
  std::mutex log_mutex_;
  std::ofstream log_file_;
  int log_pending_lines_ = 0;
  std::map<std::string, double> event_log_last_stamp_;

  std::map<int, UwbAnchor> anchors_;
  std::map<int, UwbAnchor> configured_anchors_;
  std::vector<int> anchor_order_;
  std::map<int, std::deque<UwbAnchorSample>> anchor_samples_;
  std::vector<UwbAnchorDistanceConstraint> anchor_distance_constraints_;
  std::vector<UwbRangeMeasurement> replay_measurements_;
  size_t replay_index_ = 0;
  bool replay_started_ = false;
  double replay_file_start_stamp_ = 0.0;
  bool replay_file_start_stamp_ready_ = false;
  double replay_ros_start_stamp_ = 0.0;
};

typedef std::shared_ptr<UwbManager> UwbManagerPtr;

#endif // UWB_MANAGER_H
