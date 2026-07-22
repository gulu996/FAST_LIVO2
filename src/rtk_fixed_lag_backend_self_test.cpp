#include "rtk_fixed_lag_backend.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace fast_livo_backend {

struct RtkFixedLagBackendSelfTestAccess {
  struct BoundaryResult {
    bool alignment_ready = false;
    std::set<std::int64_t> factor_stamps;
    std::size_t alignment_pending = 0;
    std::size_t graph_pending = 0;
    std::size_t alignment_pair_count = 0;
    std::size_t graph_gnss_factor_count = 0;
    std::uint64_t moved_to_graph_pending = 0;
    std::uint64_t transition_rejected = 0;
    std::uint64_t transition_waiting = 0;
    std::uint64_t terminal_rejected = 0;
    std::uint64_t silent_drop_count = 0;
    std::uint64_t duplicate_factor_count = 0;
    std::int64_t alignment_cutoff_stamp_ns = -1;
    std::int64_t conservation_delta = 0;
  };

  static void initializeTestGraph(
      RtkFixedLagBackend &backend,
      const RtkFixedLagBackend::RawOdomSample &sample) {
    gtsam::ISAM2Params parameters;
    parameters.findUnusedFactorSlots = true;
    backend.smoother_.reset(new gtsam::IncrementalFixedLagSmoother(
        backend.config_.lag_seconds, parameters));

    const gtsam::Key key = gtsam::Symbol('x', 0);
    const gtsam::Pose3 map_pose =
        backend.initial_map_to_odom_.compose(sample.pose);
    gtsam::Vector6 sigmas;
    sigmas.setConstant(0.1);
    const auto noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    gtsam::NonlinearFactorGraph factors;
    factors.add(gtsam::PriorFactor<gtsam::Pose3>(key, map_pose, noise));
    gtsam::Values values;
    values.insert(key, map_pose);
    gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
    timestamps[key] = sample.stamp.toSec();
    if (!backend.updateSmoother(factors, values, timestamps))
      throw std::runtime_error("test graph initialization failed");

    backend.keyframes_.push_back(RtkFixedLagBackend::Keyframe{
        0, key, sample.stamp, sample.pose, map_pose, true});
    backend.next_keyframe_id_ = 1;
    backend.total_nodes_created_ = 1;
    backend.initialized_ = true;
  }

  static BoundaryResult runAlignmentBoundaryOrder(bool gnss_first) {
    constexpr std::int64_t kSecondNs = 1000000000LL;
    const std::int64_t first_stamp_ns = 10 * kSecondNs;
    const std::int64_t cutoff_stamp_ns =
        first_stamp_ns + 200000000LL;
    const std::int64_t boundary_stamp_ns =
        first_stamp_ns + 300000000LL;
    const auto stamp = [](std::int64_t nanoseconds) {
      ros::Time value;
      value.fromNSec(static_cast<std::uint64_t>(nanoseconds));
      return value;
    };

    RtkFixedLagBackend backend;
    backend.config_.enable = false;
    backend.config_.save_results = false;
    backend.config_.save_text_log = false;
    backend.config_.alignment_min_pairs = 3;
    backend.config_.alignment_min_baseline_m = 2.0;
    backend.config_.alignment_max_rmse_m = 0.01;
    for (int index = 0; index < 3; ++index) {
      const std::int64_t pair_stamp_ns =
          first_stamp_ns + index * 100000000LL;
      const gtsam::Point3 position(static_cast<double>(index), 0.0, 0.0);
      backend.alignment_pairs_.push_back(
          AlignmentPair{position, position, pair_stamp_ns});
      backend.raw_odom_buffer_.push_back(RtkFixedLagBackend::RawOdomSample{
          stamp(pair_stamp_ns), gtsam::Pose3(gtsam::Rot3(), position)});
    }
    backend.alignment_gnss_used_ = backend.alignment_pairs_.size();
    backend.gnss_received_ = backend.alignment_pairs_.size();

    gtsam::Vector3 sigmas;
    sigmas.setConstant(0.1);
    const RtkFixedLagBackend::GnssMeasurement boundary{
        stamp(boundary_stamp_ns), gtsam::Point3(3.0, 0.0, 0.0), sigmas};
    if (gnss_first) {
      backend.pending_alignment_gnss_.push_back(boundary);
      ++backend.gnss_received_;
    } else {
      backend.raw_odom_buffer_.push_back(RtkFixedLagBackend::RawOdomSample{
          stamp(boundary_stamp_ns),
          gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(3.0, 0.0, 0.0))});
    }

    BoundaryResult result;
    result.alignment_ready = backend.tryFinishAlignment();
    if (!result.alignment_ready)
      throw std::runtime_error("test alignment did not finish");
    if (gnss_first) {
      backend.raw_odom_buffer_.push_back(RtkFixedLagBackend::RawOdomSample{
          stamp(boundary_stamp_ns),
          gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(3.0, 0.0, 0.0))});
    } else {
      backend.insertPendingGnss(boundary);
      ++backend.gnss_received_;
    }

    gtsam::Pose3 boundary_pose;
    double interval_s = 0.0;
    std::string interpolation_reason;
    if (!backend.interpolateRawPose(boundary.stamp, &boundary_pose, &interval_s,
                                    &interpolation_reason)) {
      throw std::runtime_error("boundary raw interpolation failed: " +
                               interpolation_reason);
    }
    initializeTestGraph(
        backend, RtkFixedLagBackend::RawOdomSample{boundary.stamp,
                                                   boundary_pose});
    if (backend.pending_factor_gnss_.size() != 1)
      throw std::runtime_error("boundary GNSS was not uniquely graph-ready");

    const std::uint64_t rejected_before = backend.gnss_rejected_;
    const bool factor_added = backend.addGnssFactor(
        backend.pending_factor_gnss_.front(), backend.keyframes_.front());
    if (factor_added) {
      result.factor_stamps.insert(backend.last_added_gnss_factor_stamp_ns_);
    } else {
      result.terminal_rejected = backend.gnss_rejected_ - rejected_before;
    }
    backend.last_processed_gnss_stamp_ns_ = boundary_stamp_ns;
    backend.pending_factor_gnss_.pop_front();
    for (const auto &factor : backend.smoother_->getFactors()) {
      if (factor && boost::dynamic_pointer_cast<GnssPositionArmFactor>(factor))
        ++result.graph_gnss_factor_count;
    }

    result.alignment_pending = backend.pending_alignment_gnss_.size();
    result.graph_pending = backend.pending_factor_gnss_.size();
    result.alignment_pair_count = backend.alignment_.pair_count;
    result.moved_to_graph_pending =
        backend.alignment_transition_to_graph_pending_;
    result.transition_rejected = backend.alignment_transition_rejected_;
    result.transition_waiting = backend.alignment_transition_waiting_;
    result.silent_drop_count = backend.gnssSilentDropCount();
    result.duplicate_factor_count = backend.gnss_duplicate_factor_count_;
    result.alignment_cutoff_stamp_ns =
        backend.alignment_last_used_gnss_stamp_ns_;
    result.conservation_delta = backend.gnssConservationDelta();
    return result;
  }
};

}  // namespace fast_livo_backend

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void testAlignment() {
  constexpr double yaw = 0.4;
  const gtsam::Rot3 rotation = gtsam::Rot3::Rz(yaw);
  const gtsam::Point3 translation(12.0, -3.0, 1.5);
  std::vector<fast_livo_backend::AlignmentPair> pairs;
  for (int i = 0; i < 30; ++i) {
    const gtsam::Point3 odom(0.5 * i, std::sin(0.2 * i), 0.05 * i);
    pairs.push_back({odom, rotation.rotate(odom) + translation});
  }
  const auto result =
      fast_livo_backend::RtkFixedLagBackend::estimateSe2Alignment(pairs);
  require(result.valid, "alignment must be observable");
  require(std::abs(result.yaw_rad - yaw) < 1e-10,
          "alignment yaw is wrong");
  require((result.translation - translation).norm() < 1e-10,
          "alignment translation is wrong");
  require(result.rmse_m < 1e-10, "alignment RMSE is wrong");
  require(result.baseline_m > 10.0, "alignment baseline is wrong");
}

void testKeyframeSelection() {
  fast_livo_backend::BackendConfig config;
  const gtsam::Pose3 origin;
  require(!fast_livo_backend::RtkFixedLagBackend::shouldCreateKeyframe(
              origin, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.1, 0, 0)),
              0.1, config),
          "small motion must not create a keyframe");
  require(fast_livo_backend::RtkFixedLagBackend::shouldCreateKeyframe(
              origin, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.8, 0, 0)),
              0.1, config),
          "translation threshold must create a keyframe");
  require(fast_livo_backend::RtkFixedLagBackend::shouldCreateKeyframe(
              origin,
              gtsam::Pose3(
                  gtsam::Rot3::Rz(9.0 / 180.0 * 3.14159265358979323846),
                  gtsam::Point3()),
              0.1, config),
          "rotation threshold must create a keyframe");
  require(fast_livo_backend::RtkFixedLagBackend::shouldCreateKeyframe(
              origin, origin, 1.0, config),
          "time threshold must create a keyframe");
}

void testLeverArmFactor() {
  const gtsam::Pose3 pose(
      gtsam::Rot3::Rz(3.14159265358979323846 / 2.0),
      gtsam::Point3(1.0, 2.0, 3.0));
  const gtsam::Point3 lever_arm(1.0, 0.0, 0.0);
  const gtsam::Point3 antenna(1.0, 3.0, 3.0);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  fast_livo_backend::GnssPositionArmFactor factor(
      gtsam::Symbol('x', 0), antenna, lever_arm, noise);
  gtsam::Matrix jacobian;
  require(factor.evaluateError(pose, jacobian).norm() < 1e-12,
          "lever arm prediction is wrong");
  require(jacobian.rows() == 3 && jacobian.cols() == 6,
          "lever arm Jacobian dimensions are wrong");
}

void testRawPoseInterpolation() {
  const ros::Time stamp0(10, 0);
  const ros::Time stamp1(10, 100000000);
  const ros::Time target(10, 50000000);
  const gtsam::Pose3 pose0(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0));
  const gtsam::Pose3 pose1(
      gtsam::Rot3::Rz(3.14159265358979323846 / 2.0),
      gtsam::Point3(2.0, 4.0, 6.0));
  gtsam::Pose3 interpolated;
  double interval_s = 0.0;
  std::string reason;
  require(fast_livo_backend::RtkFixedLagBackend::interpolatePose(
              stamp0, pose0, stamp1, pose1, target, 0.15, &interpolated,
              &interval_s, &reason),
          "valid raw pose interpolation was rejected");
  require((interpolated.translation() - gtsam::Point3(1.0, 2.0, 3.0))
                  .norm() < 1e-12,
          "raw pose translation interpolation is wrong");
  require(std::abs(gtsam::Rot3::Logmap(interpolated.rotation()).z() -
                   3.14159265358979323846 / 4.0) < 1e-12,
          "raw pose SLERP is wrong");
  require(std::abs(interval_s - 0.1) < 1e-12,
          "raw pose interpolation interval is wrong");
  require(!fast_livo_backend::RtkFixedLagBackend::interpolatePose(
              stamp0, pose0, stamp1, pose1, target, 0.05, &interpolated,
              &interval_s, &reason) &&
              reason == "RAW_ODOM_INTERPOLATION_GAP_TOO_LARGE",
          "oversized interpolation gap was not rejected precisely");
}

void testAlignmentBoundaryTransition() {
  constexpr std::int64_t kSecondNs = 1000000000LL;
  const std::int64_t cutoff_stamp_ns =
      10 * kSecondNs + 200000000LL;
  const std::int64_t boundary_stamp_ns =
      10 * kSecondNs + 300000000LL;
  const auto gnss_first = fast_livo_backend::
      RtkFixedLagBackendSelfTestAccess::runAlignmentBoundaryOrder(true);
  const auto raw_first = fast_livo_backend::
      RtkFixedLagBackendSelfTestAccess::runAlignmentBoundaryOrder(false);

  require(gnss_first.alignment_ready && raw_first.alignment_ready,
          "both callback orders must finish alignment");
  require(gnss_first.alignment_pair_count == 3 &&
              raw_first.alignment_pair_count == 3,
          "alignment must use the same three timestamped pairs");
  require(gnss_first.alignment_cutoff_stamp_ns == cutoff_stamp_ns &&
              raw_first.alignment_cutoff_stamp_ns == cutoff_stamp_ns,
          "alignment cutoff must come from the last actually used pair");
  require(gnss_first.alignment_pending == 0 &&
              raw_first.alignment_pending == 0 &&
              gnss_first.graph_pending == 0 && raw_first.graph_pending == 0,
          "both pending queues must be empty after terminal processing");
  require(gnss_first.factor_stamps == raw_first.factor_stamps &&
              gnss_first.factor_stamps ==
                  std::set<std::int64_t>{boundary_stamp_ns},
          "factor stamp sets must match across callback orders");
  require(gnss_first.graph_gnss_factor_count == 1 &&
              raw_first.graph_gnss_factor_count == 1 &&
              gnss_first.terminal_rejected == 0 &&
              raw_first.terminal_rejected == 0,
          "boundary GNSS must have exactly one factor terminal state");
  require(gnss_first.moved_to_graph_pending == 1 &&
              raw_first.moved_to_graph_pending == 0 &&
              gnss_first.transition_waiting == 1 &&
              raw_first.transition_waiting == 0,
          "alignment transition counters are wrong");
  require(gnss_first.transition_rejected == 0 &&
              raw_first.transition_rejected == 0 &&
              gnss_first.silent_drop_count == 0 &&
              raw_first.silent_drop_count == 0 &&
              gnss_first.conservation_delta == 0 &&
              raw_first.conservation_delta == 0,
          "silent_drop_count must be zero in both callback orders");
  require(gnss_first.duplicate_factor_count == 0 &&
              raw_first.duplicate_factor_count == 0,
          "duplicate_factor_count must be zero");
}

void testTrueFixedLagMarginalization() {
  gtsam::IncrementalFixedLagSmoother smoother(2.0);
  gtsam::Vector6 sigmas;
  sigmas.setConstant(0.1);
  const auto noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

  for (int i = 0; i <= 10; ++i) {
    const gtsam::Key key = gtsam::Symbol('x', i);
    const gtsam::Pose3 pose(gtsam::Rot3(), gtsam::Point3(i, 0.0, 0.0));
    gtsam::NonlinearFactorGraph factors;
    if (i == 0) {
      factors.add(gtsam::PriorFactor<gtsam::Pose3>(key, pose, noise));
    } else {
      factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
          gtsam::Symbol('x', i - 1), key,
          gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)),
          noise));
    }
    gtsam::Values values;
    values.insert(key, pose);
    gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
    timestamps[key] = i;
    smoother.update(factors, values, timestamps);
  }

  require(smoother.timestamps().size() <= 3,
          "fixed-lag state count grew beyond the 2-second window");
  for (const auto &entry : smoother.timestamps())
    require(entry.second >= 8.0,
            "fixed-lag smoother retained an expired variable");
}

}  // namespace

int main() {
  ros::Time::init();
  try {
    testAlignment();
    testKeyframeSelection();
    testLeverArmFactor();
    testRawPoseInterpolation();
    testAlignmentBoundaryTransition();
    testTrueFixedLagMarginalization();
  } catch (const std::exception &error) {
    std::cerr << "rtk_fixed_lag_backend_self_test: FAIL: " << error.what()
              << std::endl;
    return 1;
  }
  std::cout << "rtk_fixed_lag_backend_self_test: PASS" << std::endl;
  return 0;
}
