#include "common_lib.h"

#include <atomic>
#include <cassert>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define private public
#include "gnss_manager.h"
#undef private

namespace
{
std::string makeLine(int seq, double lat, double lon, double alt, int state)
{
  std::ostringstream oss;
  oss << "@IMUGNSS:{\"seq\":" << seq
      << ",\"pitch\":1.236,\"roll\":-0.418,\"yaw\":72.315"
      << ",\"lon\":" << std::fixed << std::setprecision(8) << lon
      << ",\"lat\":" << lat
      << ",\"alt\":" << std::setprecision(3) << alt
      << ",\"ve\":0.120,\"vn\":0.030,\"vu\":-0.010"
      << ",\"state\":" << state << "}";
  return oss.str();
}

std::string makeNmea(const std::string &body)
{
  uint8_t checksum = 0;
  for (char c : body) checksum ^= static_cast<uint8_t>(c);
  std::ostringstream oss;
  oss << '$' << body << '*'
      << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
      << static_cast<int>(checksum);
  return oss.str();
}

void configureTestManager(GnssManager &manager)
{
  manager.en_ = true;
  manager.update_en_ = true;
  manager.match_threshold_s_ = 10.0;
  manager.stale_timeout_s_ = 10.0;
  manager.startup_convergence_s_ = 30.0;
  manager.fixed_confirm_count_ = 2;
  manager.reacquire_confirm_count_ = 1;
  manager.fixed_state_values_ = {4};
  manager.float_state_values_ = {5};
  manager.invalid_state_values_ = {0};
  manager.ksxt_invalid_quality_values_ = {0};
  manager.ksxt_single_quality_values_ = {1};
  manager.ksxt_float_quality_values_ = {2};
  manager.ksxt_fixed_quality_values_ = {4};
  manager.gga_invalid_quality_values_ = {0};
  manager.gga_single_quality_values_ = {1};
  manager.gga_differential_quality_values_ = {2};
  manager.gga_fixed_quality_values_ = {4};
  manager.gga_float_quality_values_ = {5};
  manager.agrica_invalid_position_types_ = {0};
  manager.agrica_single_position_types_ = {1};
  manager.agrica_differential_position_types_ = {2};
  manager.agrica_fixed_position_types_ = {4};
  manager.agrica_float_position_types_ = {5};
  manager.agrica_manual_fixed_position_types_ = {7};
  manager.parser_mode_ = "auto";
  manager.primary_position_message_ = "KSXT";
  manager.fallback_position_message_ = "GGA";
  manager.fixed_only_ = true;
  manager.origin_mode_ = "first_fixed";
  manager.altitude_type_ = "ellipsoid";
  manager.frame_align_en_ = false;
  manager.frame_aligned_ = true;
  manager.frame_align_yaw_rad_ = 0.0;
  manager.frame_align_t_.setZero();
  manager.update_xy_only_ = true;
  manager.update_z_ = false;
  manager.update_orientation_ = false;
  manager.sigma_xy_fixed_m_ = 0.10;
  manager.position_cov_floor_m_ = 0.20;
  manager.chi2_gate_2d_ = 1.0e9;
  manager.max_residual_m_ = 3.0;
  manager.max_update_step_m_ = 0.20;
  manager.pause_map_update_frames_ = 3;
  manager.pause_map_update_min_correction_m_ = 0.05;
  manager.log_flush_stride_ = 1;
}

GnssUpdateResult feed(GnssManager &manager, StatesGroup &state, int seq,
                      double stamp, double lat, double lon, int device_state)
{
  manager.handleLine(makeLine(seq, lat, lon, 16.0, device_state), stamp);
  return manager.applyPositionUpdateAt(state, stamp, stamp);
}
} // namespace

int main(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  ros::Time::init();

  const std::string sample =
      "@IMUGNSS:{\"seq\":1382,\"pitch\":1.236,\"roll\":-0.418,\"yaw\":72.315,"
      "\"lon\":120.15516000,\"lat\":30.27413000,\"alt\":16.000,"
      "\"ve\":0.120,\"vn\":0.030,\"vu\":-0.010,\"state\":4}";
  const auto parsed = parseImuGnssJsonLine(sample, 123.0);
  assert(parsed.size() == 1);
  assert(parsed.front().valid);
  assert(parsed.front().seq == 1382);
  assert(std::fabs(parsed.front().latitude_deg - 30.27413) < 1e-9);
  assert(std::fabs(parsed.front().longitude_deg - 120.15516) < 1e-9);
  assert(std::fabs(parsed.front().altitude_m - 16.0) < 1e-9);
  assert(parsed.front().state == 4);

  GnssManager parser;
  configureTestManager(parser);
  const std::string ksxt_line = makeNmea(
      "KSXT,20210906104914.00,120.15516000,30.27413000,16.0000,"
      "72.315,1.236,72.315,1.000,-0.418,4,4,12,12,0,0,0,0.1,0.0,0.0,50,50");
  const auto ksxt = parser.parseLine(ksxt_line, 10.0);
  assert(ksxt.size() == 1);
  assert(ksxt.front().source_message == "KSXT");
  assert(ksxt.front().checksum_valid);
  assert(ksxt.front().device_time_valid);
  assert(ksxt.front().raw_position_quality == 4);
  assert(ksxt.front().solution_type == GnssSolutionType::RTK_FIXED);
  assert(ksxt.front().valid);
  assert(std::fabs(ksxt.front().latitude_deg - 30.27413) < 1e-9);
  assert(std::fabs(ksxt.front().longitude_deg - 120.15516) < 1e-9);

  const std::string gga_line = makeNmea(
      "GNGGA,104914.00,3016.447800,N,12009.309600,E,4,12,0.8,16.0,M,0.0,M,0.0,0000");
  const auto gga = parser.parseLine(gga_line, 10.0);
  assert(gga.size() == 1);
  assert(gga.front().source_message == "GGA");
  assert(gga.front().checksum_valid);
  assert(gga.front().raw_position_quality == 4);
  assert(gga.front().solution_type == GnssSolutionType::RTK_FIXED);
  assert(gga.front().valid);
  assert(std::fabs(gga.front().latitude_deg - 30.27413) < 1e-9);
  assert(std::fabs(gga.front().longitude_deg - 120.15516) < 1e-9);

  const auto glued = parser.parseLine("noise" + ksxt_line + gga_line, 10.0);
  assert(glued.size() == 2);
  assert(glued[0].source_message == "KSXT");
  assert(glued[1].source_message == "GGA");

  {
    GnssManager manager;
    configureTestManager(manager);
    manager.startup_convergence_s_ = 0.0;
    manager.fixed_confirm_count_ = 1;
    StatesGroup state;
    manager.handleLine(gga_line, 1.0);
    manager.handleLine(ksxt_line, 1.01);
    const GnssUpdateResult selected = manager.applyPositionUpdateAt(state, 1.01, 1.01);
    assert(selected.state_updated);
    assert(selected.source_message == "KSXT");
  }

  {
    GnssManager manager;
    configureTestManager(manager);
    manager.startup_convergence_s_ = 0.0;
    manager.fixed_confirm_count_ = 1;
    const std::string ksxt_float_line = makeNmea(
        "KSXT,20210906104914.00,120.15516000,30.27413000,16.0000,"
        "72.315,1.236,72.315,1.000,-0.418,2,4,12,12,0,0,0,0.1,0.0,0.0,50,50");
    StatesGroup state;
    manager.handleLine(gga_line, 2.0);
    manager.handleLine(ksxt_float_line, 2.01);
    const GnssUpdateResult selected = manager.applyPositionUpdateAt(state, 2.01, 2.01);
    assert(selected.state_updated);
    assert(selected.source_message == "GGA");
  }

  const V3D ecef = geodeticToEcef(0.0, 0.0, 0.0);
  assert(std::fabs(ecef.x() - 6378137.0) < 1e-6);
  assert(std::fabs(ecef.y()) < 1e-6);
  assert(std::fabs(ecef.z()) < 1e-6);
  const V3D enu0 = ecefToEnu(ecef, ecef, 0.0, 0.0);
  assert(enu0.norm() < 1e-9);

  {
    GnssManager manager;
    StatesGroup state;
    const GnssUpdateResult disabled = manager.applyPositionUpdateAt(state, 1.0, 1.0);
    assert(disabled.action == "disabled");
  }

  {
    GnssManager manager;
    configureTestManager(manager);
    manager.update_en_ = false;
    manager.startup_convergence_s_ = 0.0;
    manager.fixed_confirm_count_ = 1;
    StatesGroup state;
    const GnssUpdateResult dry_run = feed(manager, state, 1, 1.0, 30.0, 120.0, 4);
    assert(dry_run.action == "dry_run");
    assert(!dry_run.state_updated);
  }

  {
    GnssManager manager;
    configureTestManager(manager);
    manager.startup_convergence_s_ = 0.0;
    manager.fixed_confirm_count_ = 1;
    StatesGroup state;
    GnssUpdateResult result = feed(manager, state, 1, 1.0, 30.0, 120.0, 4);
    assert(result.action == "update_fixed_xy");
    result = feed(manager, state, 1, 2.0, 30.0, 120.0, 4);
    assert(result.action == "reject_duplicate");
    assert(!result.state_updated);
  }

  GnssManager manager;
  configureTestManager(manager);
  StatesGroup state;
  state.cov = MD(DIM_STATE, DIM_STATE)::Identity() * 0.01;

  GnssUpdateResult result = feed(manager, state, 1, 100.0, 30.0, 120.0, 4);
  assert(result.action == "reject_not_converged");
  assert(!result.state_updated);

  result = feed(manager, state, 2, 131.0, 30.0, 120.0, 4);
  assert(result.action == "reject_not_fixed");
  assert(!result.state_updated);

  result = feed(manager, state, 3, 132.0, 30.0, 120.0, 4);
  assert(result.action == "update_fixed_xy");
  assert(result.state_updated);

  result = feed(manager, state, 4, 133.0, 30.0, 120.0, 5);
  assert(result.action == "reject_not_fixed");
  assert(!result.state_updated);
  assert(manager.convergence_state_ == GnssManager::ConvergenceState::DEGRADED);

  result = feed(manager, state, 5, 134.0, 30.0, 120.0, 4);
  assert(result.action == "update_fixed_xy");
  assert(result.state_updated);

  result = feed(manager, state, 6, 135.0, 30.0, 120.0, 0);
  assert(result.action == "reject_invalid");
  assert(!result.state_updated);

  result = feed(manager, state, 7, 136.0, 30.0001, 120.0, 4);
  assert(result.action == "reject_large_residual");
  assert(!result.state_updated);

  {
    GnssManager replay_parser;
    configureTestManager(replay_parser);
    replay_parser.startup_convergence_s_ = 30.0;
    std::ifstream in("/home/gulu/gps_imu_recv_20260713_182453.txt");
    assert(in.is_open());
    std::map<std::string, int> counts;
    int nmea_count = 0;
    int nmea_checksum_ok = 0;
    int ksxt_quality0 = 0;
    int gga_quality0 = 0;
    int rmc_status_v = 0;
    int gsa_fix_type1 = 0;
    int agrica_position_type0 = 0;
    int update_count = 0;
    std::string line;
    double stamp = 1000.0;
    while (std::getline(in, line))
    {
      const auto measurements = replay_parser.parseLine(line, stamp);
      for (const auto &m : measurements)
      {
        counts[m.source_message]++;
        if (!m.raw_line.empty() && m.raw_line.front() == '$')
        {
          nmea_count++;
          if (m.checksum_valid) nmea_checksum_ok++;
        }
        if (m.source_message == "KSXT" && m.raw_position_quality == 0) ksxt_quality0++;
        if (m.source_message == "GGA" && m.raw_position_quality == 0) gga_quality0++;
        if (m.source_message == "RMC" && m.raw_position_quality == 0) rmc_status_v++;
        if (m.source_message == "GSA" && m.raw_position_quality == 1) gsa_fix_type1++;
        if (m.source_message == "AGRICA" && m.raw_position_quality == 0) agrica_position_type0++;
      }

      replay_parser.handleLine(line, stamp);
      StatesGroup replay_state;
      const GnssUpdateResult replay_result = replay_parser.applyPositionUpdateAt(replay_state, stamp, stamp);
      if (replay_result.state_updated) update_count++;
      stamp += 0.01;
    }
    assert(counts["KSXT"] == 67);
    assert(counts["GGA"] == 67);
    assert(counts["RMC"] == 67);
    assert(counts["GSA"] == 67);
    assert(counts["GST"] == 67);
    assert(counts["AGRICA"] == 67);
    assert(counts["ZDA"] == 7);
    assert(nmea_count == nmea_checksum_ok);
    assert(ksxt_quality0 == 67);
    assert(gga_quality0 == 67);
    assert(rmc_status_v == 67);
    assert(gsa_fix_type1 == 67);
    assert(agrica_position_type0 == 67);
    assert(update_count == 0);
    assert(!replay_parser.have_first_valid_stamp_);
  }

  GnssManager serial_manager;
  configureTestManager(serial_manager);
  serial_manager.serial_port_ = "/dev/fast_livo2_gnss_missing_for_self_test";
  assert(!serial_manager.openSerial());
  assert(!serial_manager.serial_opened_.load());

  return 0;
}
