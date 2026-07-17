#include "uwb_manager.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

int main()
{
  double corrected_range_m = 0.0;
  UwbRejectReason range_reject_reason = UwbRejectReason::NONE;
  assert(correctUwbRangeValue(8.476, 0.316, corrected_range_m, range_reject_reason));
  assert(std::fabs(corrected_range_m - 8.160) < 1e-12);
  assert(range_reject_reason == UwbRejectReason::NONE);
  assert(correctUwbRangeValue(8.476, 0.0, corrected_range_m, range_reject_reason));
  assert(corrected_range_m == 8.476);
  assert(correctUwbRangeValue(8.476, -0.1, corrected_range_m, range_reject_reason));
  assert(std::fabs(corrected_range_m - 8.576) < 1e-12);
  assert(!correctUwbRangeValue(0.0, 0.0, corrected_range_m, range_reject_reason));
  assert(range_reject_reason == UwbRejectReason::INVALID_RAW_RANGE);
  assert(!correctUwbRangeValue(0.2, 0.3, corrected_range_m, range_reject_reason));
  assert(range_reject_reason == UwbRejectReason::INVALID_CORRECTED_RANGE);
  assert(!correctUwbRangeValue(std::numeric_limits<double>::infinity(), 0.0,
                               corrected_range_m, range_reject_reason));
  assert(range_reject_reason == UwbRejectReason::INVALID_RAW_RANGE);
  assert(selectUwbPositionCovFloor(1e-5, 3.0, true, false) == 1e-5);
  assert(selectUwbPositionCovFloor(1e-5, 3.0, true, true) == 3.0);
  assert(selectUwbPositionCovFloor(0.0, 0.0, true, false) == 0.0);

  UwbRangeMeasurement debug_measurement;
  debug_measurement.anchor_id = 1;
  debug_measurement.raw_range_m = 8.476;
  debug_measurement.stamp = 10.0;
  debug_measurement.source_format = "uwbdbg";
  UwbRangeMeasurement distance_measurement = debug_measurement;
  distance_measurement.stamp = 10.012;
  distance_measurement.source_format = "distance";
  assert(isUwbReplayCrossFormatDuplicate(debug_measurement, distance_measurement, 0.05));
  distance_measurement.stamp = 10.2;
  assert(!isUwbReplayCrossFormatDuplicate(debug_measurement, distance_measurement, 0.05));

  Eigen::MatrixXd gain = Eigen::MatrixXd::Ones(DIM_STATE, 2);
  gain.row(3) << 1.0, 2.0;
  gain.row(4) << 3.0, 4.0;
  const V3D baseline_direction(3.0, 4.0, 0.0);
  const Eigen::MatrixXd used_gain =
      applyUwbUpdateMaskAndProjection(gain, false, false, &baseline_direction);

  for (int row = 0; row < DIM_STATE; ++row)
  {
    if (row == 3 || row == 4) continue;
    assert(used_gain.row(row).isZero(0.0));
  }
  const V3D unit_direction = baseline_direction.normalized();
  const Eigen::RowVectorXd cross =
      -unit_direction.y() * used_gain.row(3) + unit_direction.x() * used_gain.row(4);
  assert(cross.isZero(1e-15));

  const Eigen::Vector2d residual(0.5, -0.2);
  const Eigen::VectorXd dx = used_gain * residual;
  for (int row = 0; row < DIM_STATE; ++row)
  {
    if (row == 3 || row == 4) continue;
    assert(dx(row) == 0.0);
  }

  const Eigen::MatrixXd prior = Eigen::MatrixXd::Identity(DIM_STATE, DIM_STATE);
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(2, DIM_STATE);
  h(0, 3) = 1.0;
  h(1, 4) = 1.0;
  const Eigen::Matrix2d r = Eigen::Matrix2d::Identity() * 0.1;
  Eigen::MatrixXd updated;
  double max_asymmetry = 0.0;
  double min_diagonal = 0.0;
  assert(computeUwbJosephCovariance(prior, h, r, used_gain, updated,
                                    max_asymmetry, min_diagonal));
  const Eigen::MatrixXd i_kh = Eigen::MatrixXd::Identity(DIM_STATE, DIM_STATE) - used_gain * h;
  const Eigen::MatrixXd expected = i_kh * prior * i_kh.transpose() + used_gain * r * used_gain.transpose();
  assert(updated.isApprox(0.5 * (expected + expected.transpose()), 1e-14));
  assert(updated.allFinite());
  assert(max_asymmetry < 1e-12);
  assert(min_diagonal >= -1e-10);

  const double direct_alpha = 0.05;
  const double direct_measurement_variance = 0.25;
  Eigen::MatrixXd small_prior =
      Eigen::MatrixXd::Identity(DIM_STATE, DIM_STATE) * 1e-5;
  Eigen::MatrixXd baseline_h = Eigen::MatrixXd::Zero(1, DIM_STATE);
  baseline_h(0, 3) = 1.0;
  Eigen::MatrixXd direct_gain = Eigen::MatrixXd::Zero(DIM_STATE, 1);
  direct_gain(3, 0) = direct_alpha;
  Eigen::MatrixXd direct_updated;
  assert(computeUwbJosephCovariance(
      small_prior, baseline_h,
      Eigen::MatrixXd::Constant(1, 1, direct_measurement_variance),
      direct_gain, direct_updated, max_asymmetry, min_diagonal));
  const double expected_direct_variance =
      (1.0 - direct_alpha) * (1.0 - direct_alpha) * 1e-5 +
      direct_alpha * direct_alpha * direct_measurement_variance;
  assert(std::fabs(direct_updated(3, 3) - expected_direct_variance) < 1e-15);
  assert(direct_updated(3, 3) > small_prior(3, 3));

  UwbUpdateReport report;
  report.attempt_id = 42;
  report.status = UwbUpdateStatus::UPDATED;
  report.outcome = UwbUpdateOutcome::ACCEPTED;
  report.mode = "baseline_1d";
  report.used_anchor_ids = {1, 0};
  report.system_position_before << 1.0, 2.0, 3.0;
  report.applied_position_correction << 0.05, 0.0, 0.0;
  report.system_position_after = report.system_position_before + report.applied_position_correction;
  report.correction_norm = report.applied_position_correction.norm();
  report.state_updated = true;
  const std::string updated_line = formatUwbResultLine(report);
  assert(updated_line.find("[UWB_RESULT] attempt=42 status=UPDATED outcome=ACCEPTED") == 0);
  assert((report.system_position_after - report.system_position_before)
             .isApprox(report.applied_position_correction, 0.0));

  report.status = UwbUpdateStatus::NOT_UPDATED;
  report.outcome = UwbUpdateOutcome::REJECTED;
  report.primary_reason = UwbRejectReason::TWO_ANCHOR_RESIDUAL_GATE;
  report.state_updated = false;
  report.system_position_after = report.system_position_before;
  report.applied_position_correction.setZero();
  report.correction_norm = 0.0;
  const std::string rejected_line = formatUwbResultLine(report);
  assert(rejected_line.find("status=NOT_UPDATED outcome=REJECTED") != std::string::npos);
  assert(rejected_line.find("reason=TWO_ANCHOR_RESIDUAL_GATE") != std::string::npos);
  assert(report.system_position_after == report.system_position_before);

  report.status = UwbUpdateStatus::WAITING_INITIALIZATION;
  report.outcome = UwbUpdateOutcome::WAITING;
  report.primary_reason = UwbRejectReason::BASELINE_NOT_INITIALIZED;
  report.current_motion_m = 16.0;
  report.required_motion_m = 20.0;
  const std::string waiting_line = formatUwbResultLine(report);
  assert(waiting_line.find("status=WAITING_INITIALIZATION outcome=WAITING") != std::string::npos);
  assert(waiting_line.find("current_motion=16.000m required_motion=20.000m") != std::string::npos);

  report.status = UwbUpdateStatus::NOT_UPDATED;
  report.outcome = UwbUpdateOutcome::SKIPPED;
  report.primary_reason = UwbRejectReason::NOT_ENOUGH_VALID_ANCHORS;
  report.received_anchor_ids = {0, 1};
  report.used_anchor_ids = {1};
  report.rejected_anchor_ids = {0};
  report.required_anchor_count = 2;
  report.valid_anchor_count = 1;
  const std::string insufficient_line = formatUwbResultLine(report);
  assert(insufficient_line.find("reason=NOT_ENOUGH_VALID_ANCHORS required=2 valid=1") !=
         std::string::npos);
  const std::vector<std::string> ordered_attempt_lines = {
      insufficient_line, "[UWB_RANGE] attempt=42 anchor=0 reject_reason=RANGE_RESIDUAL_GATE"};
  assert(ordered_attempt_lines.front().find("[UWB_RESULT]") == 0);
  assert(ordered_attempt_lines.at(1).find("[UWB_RANGE]") == 0);

  const V3D anchor_a(0.0, 0.0, 0.0);
  const V3D anchor_b(3.0, 4.0, 0.0);
  assert(std::fabs((anchor_b - anchor_a).norm() - 5.0) < 1e-15);
  assert(std::fabs((anchor_b - anchor_a).normalized().norm() - 1.0) < 1e-15);
  const V3D configured_offset(0.1, -0.2, 0.3);
  const V3D estimated_add(0.02, 0.01, -0.03);
  const V3D final_offset = configured_offset + estimated_add;
  assert(final_offset.isApprox(V3D(0.12, -0.19, 0.27), 1e-15));

  std::cout << "uwb_manager_self_test: PASS\n";
  return 0;
}
