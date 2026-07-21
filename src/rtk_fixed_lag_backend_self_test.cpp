#include "rtk_fixed_lag_backend.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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
  try {
    testAlignment();
    testKeyframeSelection();
    testLeverArmFactor();
    testRawPoseInterpolation();
    testTrueFixedLagMarginalization();
  } catch (const std::exception &error) {
    std::cerr << "rtk_fixed_lag_backend_self_test: FAIL: " << error.what()
              << std::endl;
    return 1;
  }
  std::cout << "rtk_fixed_lag_backend_self_test: PASS" << std::endl;
  return 0;
}
