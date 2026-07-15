/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#ifndef GNSS_MANAGER_H
#define GNSS_MANAGER_H

#include "common_lib.h"

#include <atomic>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

enum class GnssSolutionType
{
  INVALID = 0,
  SINGLE,
  DIFFERENTIAL,
  RTK_FLOAT,
  RTK_FIXED,
  MANUAL_FIXED,
  UNKNOWN
};

struct GnssMeasurement
{
  int seq = -1;
  double stamp = 0.0;
  double receive_stamp = 0.0;
  double device_stamp = 0.0;
  bool device_time_valid = false;

  double latitude_deg = 0.0;
  double longitude_deg = 0.0;
  double altitude_m = 0.0;

  double roll_deg = 0.0;
  double pitch_deg = 0.0;
  double yaw_deg = 0.0;

  double velocity_east = 0.0;
  double velocity_north = 0.0;
  double velocity_up = 0.0;

  int state = 0;
  GnssSolutionType solution_type = GnssSolutionType::INVALID;
  int raw_position_quality = 0;
  int raw_heading_quality = 0;
  int satellite_count = 0;
  double hdop = std::numeric_limits<double>::quiet_NaN();
  double horizontal_std_m = std::numeric_limits<double>::quiet_NaN();
  double vertical_std_m = std::numeric_limits<double>::quiet_NaN();
  double differential_age_s = std::numeric_limits<double>::quiet_NaN();
  std::string source_message;
  std::string raw_line;
  std::string reject_reason;

  bool checksum_valid = false;
  bool valid = false;
};

struct GnssUpdateResult
{
  bool state_updated = false;
  bool request_pause_map_insert = false;
  std::string action = "none";

  int seq = -1;
  int device_state = 0;
  std::string source_message;
  std::string convergence_state;
  double stamp = 0.0;
  double time_diff_s = 0.0;
  double residual_norm = 0.0;
  double mahalanobis_distance = 0.0;
  double correction_norm = 0.0;
  double correction_raw_norm = 0.0;
  double correction_applied_norm = 0.0;
  double sigma_xy = 0.0;
  int pause_map_update_frames = 0;
  double pause_map_update_min_correction_m = 0.0;

  V3D enu_position = V3D::Zero();
  V3D world_position = V3D::Zero();
  V3D predicted_position = V3D::Zero();
  V3D residual = V3D::Zero();
  V3D correction_raw = V3D::Zero();
  V3D correction_applied = V3D::Zero();
};

struct GnssFrameAlignSample
{
  V3D enu_position = V3D::Zero();
  V3D world_position = V3D::Zero();
  double stamp = 0.0;
};

V3D geodeticToEcef(double latitude_deg, double longitude_deg, double ellipsoid_height_m);
V3D ecefToEnu(const V3D &ecef, const V3D &origin_ecef,
              double origin_latitude_deg, double origin_longitude_deg);
std::vector<GnssMeasurement> parseImuGnssJsonLine(const std::string &line, double stamp);

class GnssManager
{
public:
  GnssManager();
  ~GnssManager();

  bool initialize(ros::NodeHandle &nh, const std::string &save_path);
  void shutdown();
  bool updateEnabled() const { return en_; }

  GnssUpdateResult applyPositionUpdateAt(StatesGroup &state,
                                         double current_lidar_stamp,
                                         double lidar_start_stamp);

private:
  enum class ConvergenceState
  {
    DISABLED = 0,
    SERIAL_OPENED,
    WAIT_VALID_DATA,
    WARMING_UP,
    WAIT_FIXED,
    ALIGNING,
    READY,
    DEGRADED
  };

  bool loadParameters(ros::NodeHandle &nh);
  bool openSerial();
  bool configureSerial();
  void closeSerial();
  void readLoop();
  void handleLine(const std::string &line, double stamp);
  std::vector<GnssMeasurement> parseLine(const std::string &line, double receive_stamp) const;
  bool takeLatestMeasurement(double current_lidar_stamp, GnssMeasurement &measurement, double &time_diff_s);

  bool isFixedState(int state) const;
  bool isFixedSolution(const GnssMeasurement &measurement) const;
  bool isInvalidState(int state) const;
  double ellipsoidHeight(const GnssMeasurement &measurement) const;
  bool ensureOrigin(const GnssMeasurement &measurement);
  bool convertMeasurement(const GnssMeasurement &measurement, V3D &enu, V3D &world);
  bool updateConvergenceAndAlignment(const GnssMeasurement &measurement,
                                     double time_diff_s,
                                     const StatesGroup &state,
                                     V3D &enu,
                                     V3D &world,
                                     std::string &reject_action);
  void collectAlignSample(const GnssMeasurement &measurement,
                          const StatesGroup &state,
                          const V3D &enu);
  bool trySolveFrameAlignment();
  bool solveFrameAlignment(const std::vector<GnssFrameAlignSample> &solve_samples,
                           const std::vector<GnssFrameAlignSample> &validation_samples,
                           double &yaw_rad, V3D &translation,
                           double &rms, double &max_error) const;
  V3D enuToWorld(const V3D &enu) const;

  GnssUpdateResult rejectResult(const GnssMeasurement &measurement,
                                const std::string &action,
                                double time_diff_s,
                                const V3D &enu,
                                const V3D &world,
                                const V3D &pred,
                                const V3D &residual,
                                double residual_norm,
                                double mahalanobis);
  void logRawLine(double stamp, const std::string &line);
  void logParsedMeasurement(const GnssMeasurement &measurement);
  void logUpdate(const GnssUpdateResult &result, const GnssMeasurement &measurement);
  void logEventThrottled(double stamp, const std::string &key, double period_s,
                         const std::string &level, const std::string &message);
  void transitionTo(ConvergenceState state, double stamp, const std::string &event);
  std::string convergenceStateName() const;

  bool en_ = false;
  bool update_en_ = true;
  std::string input_source_ = "serial";
  std::string serial_port_ = "/dev/ttyUSB0";
  int baudrate_ = 921600;
  bool dtr_high_ = true;
  bool rts_high_ = false;
  std::string parser_mode_ = "auto";
  std::string primary_position_message_ = "KSXT";
  std::string fallback_position_message_ = "GGA";
  double time_offset_s_ = 0.0;
  double match_threshold_s_ = 0.10;
  double stale_timeout_s_ = 0.50;
  int max_queue_size_ = 512;

  double startup_convergence_s_ = 30.0;
  int fixed_confirm_count_ = 10;
  int reacquire_confirm_count_ = 5;
  bool reset_convergence_on_long_stale_ = false;
  double reset_convergence_stale_s_ = 5.0;
  std::set<int> fixed_state_values_{4};
  std::set<int> float_state_values_{5};
  std::set<int> invalid_state_values_{0};
  std::set<int> ksxt_invalid_quality_values_{0};
  std::set<int> ksxt_single_quality_values_{1};
  std::set<int> ksxt_float_quality_values_{2};
  std::set<int> ksxt_fixed_quality_values_{4};
  std::set<int> gga_invalid_quality_values_{0};
  std::set<int> gga_single_quality_values_{1};
  std::set<int> gga_differential_quality_values_{2};
  std::set<int> gga_fixed_quality_values_{4};
  std::set<int> gga_float_quality_values_{5};
  std::set<int> agrica_invalid_position_types_{0};
  std::set<int> agrica_single_position_types_{1};
  std::set<int> agrica_differential_position_types_{2};
  std::set<int> agrica_fixed_position_types_{4};
  std::set<int> agrica_float_position_types_{5};
  std::set<int> agrica_manual_fixed_position_types_{7};
  bool agrica_crc_check_en_ = false;
  bool fixed_only_ = true;

  std::string origin_mode_ = "first_fixed";
  double origin_latitude_deg_ = 0.0;
  double origin_longitude_deg_ = 0.0;
  double origin_altitude_m_ = 0.0;
  std::string altitude_type_ = "ellipsoid";
  double geoid_separation_m_ = 0.0;
  bool origin_ready_ = false;
  V3D origin_ecef_ = V3D::Zero();

  bool frame_align_en_ = true;
  std::string frame_align_mode_ = "trajectory_2d";
  int frame_align_min_samples_ = 20;
  double frame_align_min_motion_m_ = 10.0;
  double frame_align_max_rms_m_ = 0.50;
  double frame_align_max_error_m_ = 1.50;
  bool frame_align_freeze_after_success_ = true;
  double frame_align_yaw_deg_ = 0.0;
  V3D frame_align_translation_ = V3D::Zero();
  bool frame_aligned_ = false;
  double frame_align_yaw_rad_ = 0.0;
  V3D frame_align_t_ = V3D::Zero();
  std::vector<GnssFrameAlignSample> frame_align_samples_;

  bool update_xy_only_ = true;
  bool update_z_ = false;
  bool update_orientation_ = false;
  double sigma_xy_fixed_m_ = 0.10;
  double sigma_z_fixed_m_ = 0.30;
  double position_cov_floor_m_ = 0.20;
  double chi2_gate_2d_ = 9.21;
  double max_residual_m_ = 3.0;
  double max_update_step_m_ = 0.20;
  V3D lever_arm_body_to_gnss_ = V3D::Zero();
  int pause_map_update_frames_ = 3;
  double pause_map_update_min_correction_m_ = 0.05;

  std::string raw_log_filename_ = "gnss_raw.txt";
  std::string parsed_log_filename_ = "gnss_parsed.txt";
  std::string update_log_filename_ = "gnss_updates.txt";
  int log_flush_stride_ = 1;

  int serial_fd_ = -1;
  std::atomic<bool> running_{false};
  std::atomic<bool> serial_opened_{false};
  std::thread read_thread_;

  mutable std::mutex measurement_mutex_;
  std::deque<GnssMeasurement> measurement_queue_;
  bool have_last_seq_ = false;
  int last_seq_ = -1;
  bool have_last_update_epoch_ = false;
  double last_update_epoch_stamp_ = 0.0;
  std::string last_update_epoch_source_;
  double last_measurement_stamp_ = 0.0;
  bool have_last_measurement_stamp_ = false;

  std::mutex state_mutex_;
  ConvergenceState convergence_state_ = ConvergenceState::DISABLED;
  bool have_first_valid_stamp_ = false;
  double first_valid_stamp_ = 0.0;
  int consecutive_fixed_count_ = 0;
  bool was_ready_once_ = false;

  std::mutex log_mutex_;
  std::ofstream raw_log_file_;
  std::ofstream parsed_log_file_;
  std::ofstream update_log_file_;
  int raw_log_pending_lines_ = 0;
  int parsed_log_pending_lines_ = 0;
  int update_log_pending_lines_ = 0;
  std::map<std::string, double> event_log_last_stamp_;
};

typedef std::shared_ptr<GnssManager> GnssManagerPtr;

#endif // GNSS_MANAGER_H
