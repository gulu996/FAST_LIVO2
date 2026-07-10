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
  double time_diff_s = 0.0;
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

struct UwbRepeatedRangeState
{
  bool valid = false;
  double last_range_m = 0.0;
  double first_stamp = 0.0;
  double last_stamp = 0.0;
  int repeat_count = 0;
};

struct UwbAnchorFrameAlignSample
{
  V3D tag_position_w = V3D::Zero();
  int anchor_id = -1;
  double range_m = 0.0;
  double stamp = 0.0;
};

struct UwbUpdateResult
{
  int used_count = 0;
  std::string action = "none";
  double residual_rms = 0.0;
  double max_abs_residual = 0.0;
  double xy_correction_raw = 0.0;
  double xy_correction_applied = 0.0;
  double time_diff = 0.0;
  double correction_norm = 0.0;
  double baseline_consistency_error = 0.0;
  V3D uwb_only_position = V3D::Zero();
  V3D filtered_uwb_position = V3D::Zero();
  double uwb_only_residual_rms = 0.0;
  double uwb_only_max_abs_residual = 0.0;
  double uwb_only_position_jump = 0.0;
  double uwb_only_speed = 0.0;
  double slam_uwb_position_diff = 0.0;
  int limited_update_consecutive_good_count = 0;
  int relocalization_candidate_count = 0;
  bool state_updated = false;
  bool request_pause_map_insert = false;
  bool request_relocalization = false;
  bool relocalization_confirmed = false;
  bool local_map_reset = false;
  bool visual_cache_reset = false;
  bool covariance_inflated = false;
};

struct UwbOnlyPositionSample
{
  V3D position = V3D::Zero();
  double stamp = 0.0;
};

class UwbManager
{
public:
  UwbManager();
  ~UwbManager();

  bool initialize(ros::NodeHandle &nh, const std::string &save_path);
  void shutdown();
  bool isRunning() const { return running_.load(); }
  bool updateEnabled() const { return en_; }
  void setDegenerateMode(bool degenerated) { degraded_mode_ = degenerated; }
  UwbUpdateResult applyRangeUpdate(StatesGroup &state);
  UwbUpdateResult applyRangeUpdateAt(StatesGroup &state, double current_lidar_stamp, double lidar_start_stamp);

private:
  bool loadParameters(ros::NodeHandle &nh);
  bool openSerial();
  bool configureSerial();
  void closeSerial();
  void readLoop();
  bool loadReplayFile();
  std::vector<UwbRangeMeasurement> takeReplayMeasurements(double current_lidar_stamp, double lidar_start_stamp);
  void handleLine(const std::string &line, double stamp);
  std::vector<UwbRangeMeasurement> parseLine(const std::string &line, double stamp) const;
  std::vector<UwbRangeMeasurement> filterRepeatedRanges(const std::vector<UwbRangeMeasurement> &measurements,
                                                        const std::string &source);
  std::vector<UwbRangeMeasurement> takeRecentMeasurements(double now);
  void logRawLine(double stamp, const std::string &line, const std::vector<UwbRangeMeasurement> &measurements);
  void logEvent(double stamp, const std::string &level, const std::string &message);
  void logEventThrottled(double stamp, const std::string &key, double period_s,
                         const std::string &level, const std::string &message);
  void logUpdate(double stamp, int used_count, double residual_norm, const V3D &rot_add,
                 const V3D &trans_add, const V3D &tag_offset_add);
  UwbUpdateResult applyLatestMeasurements(StatesGroup &state, const std::vector<UwbRangeMeasurement> &measurements);
  double effectivePositionCovFloor() const;
  bool solveUwbOnlyPosition2D(const std::vector<UwbRangeMeasurement> &measurements,
                              double z_world, const V3D &initial_position,
                              V3D &position, double &residual_rms,
                              double &max_abs_residual, double &geometry_score) const;
  V3D updateFilteredUwbOnlyPosition(const V3D &position, double stamp,
                                    double &position_jump, double &speed);
  bool tryAlignAnchorFrame(const StatesGroup &state,
                           const std::vector<UwbRangeMeasurement> &measurements);
  bool tryInitializeBaselineAnchors(const StatesGroup &state,
                                    const std::vector<UwbRangeMeasurement> &measurements);
  double configuredBaselineDistance() const;
  void collectAnchorEstimateSamples(const StatesGroup &state, const std::vector<UwbRangeMeasurement> &measurements);
  bool estimateAnchorPosition(int anchor_id, UwbAnchor &anchor, double &rmse, int &rank,
                              std::string *failure_reason = nullptr) const;
  void applyAnchorDistanceConstraints();
  void logAnchorEstimate(int anchor_id, const V3D &position_w, double rmse, int rank, int sample_count);
  bool estimateTWorldUwbByRanges(M3D &R_ext_to_w, V3D &t_ext_to_w, double &residual_rms,
                                 double &max_abs_residual, double &residual_rms_before,
                                 int &valid_range_count, double &trajectory_motion,
                                 std::vector<int> &used_anchor_ids, std::string *failure_reason) const;
  bool evaluateAnchorFrameResiduals(const std::map<int, UwbAnchor> &candidate_anchors,
                                    const std::vector<UwbAnchorFrameAlignSample> &samples,
                                    double &residual_rms, double &max_abs_residual,
                                    int &valid_range_count, std::vector<int> *used_anchor_ids = nullptr) const;

  bool en_ = false;
  bool update_en_ = true;
  std::string input_source_ = "serial";
  std::string serial_port_ = "/dev/ttyUSB0";
  int baudrate_ = 115200;
  bool dtr_high_ = true;
  bool rts_high_ = false;
  std::string mode_ = "entry_exit_distance";
  std::string parser_mode_ = "uwb";
  std::string log_filename_ = "uwb_ranges.txt";
  std::string update_log_filename_ = "uwb_updates.txt";
  int log_flush_stride_ = 1;
  std::string replay_file_;
  double replay_start_offset_s_ = 0.0;
  double replay_match_threshold_s_ = 0.05;
  double range_scale_ = 1.0;
  double min_range_m_ = 0.05;
  double max_range_m_ = 250.0;
  double max_age_s_ = 0.5;
  int max_queue_size_ = 512;
  int min_update_anchors_ = 2;
  int min_anchors_for_update_ = 3;
  int prefer_anchors_ = 3;
  double range_noise_m_ = 0.10;
  bool residual_debug_only_ = false;
  bool update_xy_only_ = true;
  bool use_3d_range_model_ = true;
  bool update_z_ = false;
  bool update_orientation_ = false;
  double max_residual_rms_ = 0.50;
  double max_xy_correction_normal_ = 0.50;
  double max_update_step_xy_ = 0.10;
  double two_anchor_sigma_scale_ = 5.0;
  std::string two_anchor_update_mode_ = "baseline_1d_direct";
  std::string two_anchor_policy_when_total_anchors_gt2_ = "dry_run";
  double baseline_1d_direct_alpha_ = 0.05;
  double baseline_1d_direct_max_step_m_ = 0.03;
  double two_anchor_normal_max_step_m_ = 0.05;
  double two_anchor_degraded_max_step_m_ = 0.10;
  double two_anchor_strong_degraded_max_step_m_ = 0.15;
  double two_anchor_hard_max_step_m_ = 0.20;
  double two_anchor_max_residual_ = 2.0;
  double two_anchor_baseline_consistency_threshold_m_ = 2.0;
  double two_anchor_max_residual_rms_ = 0.8;
  double two_anchor_max_abs_residual_ = 1.5;
  bool single_anchor_corridor_1d_en_ = true;
  bool single_anchor_only_when_total_anchors_eq_2_ = true;
  bool single_anchor_requires_baseline_initialized_ = true;
  double single_anchor_alpha_ = 0.05;
  double single_anchor_normal_max_step_m_ = 0.05;
  double single_anchor_degraded_max_step_m_ = 0.10;
  double single_anchor_strong_degraded_max_step_m_ = 0.15;
  double single_anchor_hard_max_step_m_ = 0.20;
  double single_anchor_max_residual_ = 2.5;
  int single_anchor_confirm_count_required_ = 1;
  double single_anchor_min_range_m_ = 1.0;
  double single_anchor_max_range_m_ = 60.0;
  double single_anchor_branch_margin_m_ = 0.3;
  double single_anchor_near_anchor_disable_dist_m_ = 0.3;
  double single_anchor_range_jump_threshold_m_ = 2.5;
  double single_anchor_residual_jump_threshold_m_ = 1.5;
  double single_anchor_speed_threshold_mps_ = 2.0;
  double corridor_direction_max_angle_deg_ = 45.0;
  double min_motion_for_direction_check_m_ = 0.3;
  int direction_check_window_frames_ = 10;
  bool disable_single_anchor_on_turn_ = true;
  bool enable_corridor_segments_ = false;
  bool degraded_mode_en_ = true;
  int degraded_confirm_count_ = 3;
  int strong_degraded_confirm_count_ = 5;
  double multi_anchor_max_residual_rms_ = 1.0;
  double multi_anchor_max_abs_residual_ = 1.5;
  double max_time_diff_s_ = 0.05;
  double limited_update_max_residual_rms_ = 2.0;
  double limited_update_max_abs_residual_ = 3.0;
  double limited_update_max_xy_raw_ = 2.0;
  double limited_update_max_time_diff_s_ = 0.05;
  int limited_update_require_consecutive_good_ = 2;
  int limited_update_consecutive_good_count_ = 0;
  double normal_update_max_xy_raw_ = 0.80;
  bool uwb_debug_force_limited_update_ = false;
  double relocalization_candidate_min_xy_raw_ = 1.5;
  double hard_reject_xy_raw_ = 3.0;
  bool relocalization_en_ = false;
  double relocalization_threshold_m_ = 1.5;
  int relocalization_confirm_count_ = 5;
  double uwb_only_max_residual_rms_ = 0.5;
  double uwb_only_max_abs_residual_ = 1.0;
  double uwb_position_jump_threshold_m_ = 1.0;
  double uwb_speed_threshold_mps_ = 2.0;
  double anchor_geometry_min_score_ = 1e-3;
  int relocalization_candidate_count_ = 0;
  int relocalization_candidate_count_total_ = 0;
  double relocalization_slam_diff_sum_ = 0.0;
  double relocalization_uwb_only_residual_sum_ = 0.0;
  double relocalization_uwb_only_jump_sum_ = 0.0;
  int two_anchor_update_count_ = 0;
  int two_anchor_reject_count_ = 0;
  double two_anchor_xy_correction_sum_ = 0.0;
  double two_anchor_xy_correction_max_ = 0.0;
  double two_anchor_abs_baseline_residual_sum_ = 0.0;
  double two_anchor_baseline_consistency_sum_ = 0.0;
  int corridor_residual_stable_count_ = 0;
  bool corridor_last_residual_valid_ = false;
  double corridor_last_residual_ = 0.0;
  bool corridor_last_tag_valid_ = false;
  V3D corridor_last_tag_position_w_ = V3D::Zero();
  std::deque<V3D> corridor_direction_history_;
  int single_anchor_last_anchor_id_ = -1;
  int single_anchor_last_branch_ = 0;
  int single_anchor_confirm_counter_ = 1;
  bool single_anchor_last_valid_ = false;
  double single_anchor_last_range_m_ = 0.0;
  double single_anchor_last_residual_m_ = 0.0;
  double single_anchor_last_stamp_ = 0.0;
  int require_consecutive_good_updates_ = 3;
  double good_residual_rms_ = 0.30;
  bool suspect_hold_en_ = false;
  bool lost_hold_en_ = false;
  int uwb_state_ = 0; // 0 NORMAL, 1 SUSPECT, 2 LOST.
  int uwb_consecutive_good_count_ = 0;
  int uwb_lost_good_count_ = 0;
  bool uwb_consecutive_gate_ready_ = false;
  double large_correction_warn_threshold_ = 0.50;
  double large_correction_reject_threshold_ = 3.0;
  std::string anchor_file_;
  double position_cov_floor_m_ = 0.0;
  double position_cov_floor_degraded_m_ = 3.0;
  bool position_cov_floor_degraded_only_ = false;
  bool degraded_mode_ = false;
  double max_residual_m_ = 6.0;
  bool stale_repeat_filter_en_ = true;
  double stale_repeat_epsilon_m_ = 0.001;
  int stale_repeat_max_count_ = 3;
  double stale_repeat_max_duration_s_ = 2.0;
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
  bool two_anchor_baseline_mode_ = false;
  bool anchor_frame_align_en_ = false;
  int anchor_frame_align_start_id_ = 0;
  int anchor_frame_align_end_id_ = 1;
  double anchor_frame_align_min_motion_m_ = 20.0;
  bool anchor_frame_align_use_start_range_offset_ = true;
  bool anchor_frame_align_yaw_only_ = true;
  bool anchor_frame_aligned_ = false;
  bool anchor_frame_align_failed_ = false;
  bool anchor_frame_align_candidate_ready_ = false;
  double anchor_frame_align_min_duration_s_ = 30.0;
  int anchor_frame_align_min_ranges_ = 30;
  int anchor_frame_align_min_anchors_ = 3;
  double anchor_frame_align_success_rms_m_ = 0.50;
  double anchor_frame_align_success_max_residual_m_ = 1.50;
  double anchor_frame_align_validation_duration_s_ = 5.0;
  double anchor_frame_align_validation_start_stamp_ = 0.0;
  bool anchor_frame_align_start_pose_ready_ = false;
  bool anchor_frame_align_start_range_ready_ = false;
  V3D anchor_frame_align_start_tag_position_w_ = V3D::Zero();
  double anchor_frame_align_start_range_m_ = 0.0;
  M3D anchor_frame_align_R_ext_to_w_ = M3D::Identity();
  V3D anchor_frame_align_t_ext_to_w_ = V3D::Zero();
  std::vector<UwbAnchorFrameAlignSample> anchor_frame_align_samples_;
  std::vector<UwbAnchorFrameAlignSample> anchor_frame_align_validation_samples_;
  std::map<int, UwbAnchor> pending_aligned_anchors_;

  int serial_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread read_thread_;
  mutable std::mutex measurement_mutex_;
  std::deque<UwbRangeMeasurement> measurement_queue_;
  std::mutex log_mutex_;
  std::ofstream raw_log_file_;
  std::ofstream update_log_file_;
  int raw_log_pending_lines_ = 0;
  std::map<std::string, double> event_log_last_stamp_;

  std::map<int, UwbAnchor> anchors_;
  std::map<int, UwbAnchor> configured_anchors_;
  std::vector<int> anchor_order_;
  std::map<int, std::deque<UwbAnchorSample>> anchor_samples_;
  std::vector<UwbAnchorDistanceConstraint> anchor_distance_constraints_;
  std::map<int, UwbRepeatedRangeState> repeated_range_states_;
  std::vector<UwbRangeMeasurement> replay_measurements_;
  size_t replay_index_ = 0;
  bool replay_started_ = false;
  double replay_last_slam_relative_time_ = -1.0;
  double replay_file_start_stamp_ = 0.0;
  bool replay_file_start_stamp_ready_ = false;
  std::deque<UwbOnlyPositionSample> uwb_only_position_history_;
};

typedef std::shared_ptr<UwbManager> UwbManagerPtr;

#endif // UWB_MANAGER_H
