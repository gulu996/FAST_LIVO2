/*
Small dependency-free regression check for the GNSS adapter core.
*/

#include "gnss_adapter.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void check(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

bool near(double actual, double expected, double tolerance = 1e-9)
{
  return std::fabs(actual - expected) <= tolerance;
}

GnssAdapterConfig manualConfig()
{
  GnssAdapterConfig config;
  config.origin_mode = "manual";
  config.origin_lla << 22.5299731, 113.9331154, 39.324;
  config.fixed_confirm_count = 2;
  config.fixed_lost_count = 2;
  config.recovery_confirm_count = 3;
  return config;
}

gnss_comm::GnssPVTSolnMsg fixedMessage(double tow = 186157.8)
{
  gnss_comm::GnssPVTSolnMsg message;
  message.time.week = 2393;
  message.time.tow = tow;
  message.fix_type = 3;
  message.valid_fix = true;
  message.diff_soln = true;
  message.carr_soln = 2;
  message.num_sv = 25;
  message.latitude = 22.5299731;
  message.longitude = 113.9331154;
  message.altitude = 39.324;
  message.height_msl = 30.0;
  message.h_acc = 0.014;
  message.v_acc = 0.016;
  message.p_dop = 1.63;
  message.vel_n = 1.0;
  message.vel_e = 2.0;
  message.vel_d = -0.5;
  message.vel_acc = 0.02;
  return message;
}

void testTimeStateEnuAndCovariance()
{
  GnssAdapter adapter(manualConfig());
  const ros::Time callback_time(1763437340, 0);

  const GnssAdapterResult first = adapter.process(fixedMessage(), callback_time);
  check(near(first.status.header.stamp.toSec(), 1763437339.8, 1e-6),
        "GPST to UTC conversion omitted/duplicated the leap-second correction");
  check(first.status.raw_quality == fast_livo::GnssStatus::RTK_FIXED,
        "sample must classify as raw RTK_FIXED");
  check(first.status.filtered_quality == fast_livo::GnssStatus::INVALID,
        "first fixed sample must await confirmation");
  check(!first.status.accepted && first.status.reject_reason == "RTK_NOT_CONFIRMED",
        "unconfirmed fixed sample must be rejected explicitly");

  const GnssAdapterResult second = adapter.process(fixedMessage(186157.9), callback_time);
  check(second.status.filtered_quality == fast_livo::GnssStatus::RTK_FIXED,
        "confirmed fixed sample must become active");
  check(second.status.accepted && second.publish_odometry,
        "confirmed fixed sample with manual origin must publish");
  check(std::fabs(second.odometry.pose.pose.position.x) < 1e-6 &&
        std::fabs(second.odometry.pose.pose.position.y) < 1e-6 &&
        std::fabs(second.odometry.pose.pose.position.z) < 1e-6,
        "manual origin equal to measurement must produce zero ENU");
  check(near(second.odometry.twist.twist.linear.x, 2.0) &&
        near(second.odometry.twist.twist.linear.y, 1.0) &&
        near(second.odometry.twist.twist.linear.z, 0.5),
        "NED velocity must map to ENU");
  check(near(second.odometry.pose.covariance[0], 0.03 * 0.03) &&
        near(second.odometry.pose.covariance[7], 0.03 * 0.03) &&
        near(second.odometry.pose.covariance[14], 0.05 * 0.05),
        "position standard deviations must be clamped then squared");
  check(near(second.odometry.twist.covariance[0], 0.05 * 0.05),
        "velocity standard deviation floor must be squared");
  check(second.odometry.pose.covariance[21] >= 1e6 &&
        second.odometry.pose.covariance[28] >= 1e6 &&
        second.odometry.pose.covariance[35] >= 1e6,
        "placeholder orientation must have unknown/large covariance");
}

void testInvalidInputs()
{
  const GnssAdapterConfig config = manualConfig();

  gnss_comm::GnssPVTSolnMsg message = fixedMessage();
  message.latitude = std::numeric_limits<double>::quiet_NaN();
  GnssAdapter nan_adapter(config);
  check(nan_adapter.process(message, ros::Time(1)).status.reject_reason == "INVALID_LATITUDE",
        "NaN latitude must be rejected");

  message = fixedMessage();
  message.valid_fix = false;
  GnssAdapter invalid_fix_adapter(config);
  check(invalid_fix_adapter.process(message, ros::Time(1)).status.reject_reason == "INVALID_FIX",
        "invalid fix must be rejected");

  message = fixedMessage();
  message.carr_soln = 1;
  GnssAdapter float_adapter(config);
  const GnssAdapterResult float_result = float_adapter.process(message, ros::Time(1));
  check(float_result.status.raw_quality == fast_livo::GnssStatus::RTK_FLOAT &&
        !float_result.status.accepted,
        "RTK Float must classify correctly and obey the default reject policy");

  message = fixedMessage();
  message.num_sv = 5;
  GnssAdapter satellite_adapter(config);
  check(satellite_adapter.process(message, ros::Time(1)).status.reject_reason ==
            "NUM_SV_TOO_LOW",
        "too few satellites must be rejected");

  message = fixedMessage();
  message.h_acc = 1.42;
  GnssAdapter accuracy_adapter(config);
  check(accuracy_adapter.process(message, ros::Time(1)).status.reject_reason ==
            "H_ACC_TOO_LARGE",
        "large horizontal accuracy estimate must be rejected");
}

void testTimeOrderingAndMissingOrigin()
{
  GnssAdapterConfig config = manualConfig();
  config.fixed_confirm_count = 1;
  GnssAdapter adapter(config);
  check(adapter.process(fixedMessage(), ros::Time(1)).status.accepted,
        "single-frame test configuration must activate fixed state");
  check(adapter.process(fixedMessage(), ros::Time(1)).status.reject_reason ==
            "DUPLICATE_GNSS_TIME",
        "duplicate GNSS time must not be processed twice");
  check(adapter.process(fixedMessage(186157.7), ros::Time(1)).status.reject_reason ==
            "NON_MONOTONIC_TIME",
        "backward GNSS time must be rejected");

  config.origin_mode = "first_fixed";
  config.accept_rtk_float = true;
  GnssAdapter no_origin_adapter(config);
  gnss_comm::GnssPVTSolnMsg float_message = fixedMessage();
  float_message.carr_soln = 1;
  const GnssAdapterResult no_origin = no_origin_adapter.process(float_message, ros::Time(1));
  check(!no_origin.status.origin_initialized && !no_origin.status.accepted &&
        no_origin.status.reject_reason == "ORIGIN_NOT_INITIALIZED",
        "enabled non-fixed quality must still wait for a confirmed-fixed origin");
}

void testLossRecoveryAndAverageOrigin()
{
  GnssAdapter adapter(manualConfig());
  adapter.process(fixedMessage(), ros::Time(1));
  check(adapter.process(fixedMessage(186157.9), ros::Time(1)).status.accepted,
        "state must become active after initial confirmation");

  gnss_comm::GnssPVTSolnMsg invalid = fixedMessage(186158.0);
  invalid.valid_fix = false;
  const GnssAdapterResult first_loss = adapter.process(invalid, ros::Time(1));
  check(first_loss.status.filtered_quality == fast_livo::GnssStatus::RTK_FIXED &&
        !first_loss.status.accepted && first_loss.status.consecutive_lost_count == 1,
        "ACTIVE_FIXED must hold filtered state, but reject the current bad sample");
  invalid.time.tow = 186158.1;
  adapter.process(invalid, ros::Time(1));

  GnssAdapterResult recovery = adapter.process(fixedMessage(186158.2), ros::Time(1));
  check(recovery.status.filtered_quality == fast_livo::GnssStatus::RECOVERING &&
        !recovery.status.accepted,
        "first fixed sample after loss must enter RECOVERING");
  recovery = adapter.process(fixedMessage(186158.3), ros::Time(1));
  check(!recovery.status.accepted, "recovery must require consecutive fixed samples");
  recovery = adapter.process(fixedMessage(186158.4), ros::Time(1));
  check(recovery.status.filtered_quality == fast_livo::GnssStatus::RTK_FIXED &&
        recovery.status.accepted,
        "recovery confirmation count must restore active fixed state");

  GnssAdapterConfig average_config = manualConfig();
  average_config.origin_mode = "average_fixed";
  average_config.fixed_confirm_count = 1;
  average_config.origin_average_count = 2;
  GnssAdapter average_adapter(average_config);
  check(!average_adapter.process(fixedMessage(), ros::Time(1)).status.origin_initialized,
        "average origin must wait for the configured sample count");
  gnss_comm::GnssPVTSolnMsg interrupted_float = fixedMessage(186157.9);
  interrupted_float.carr_soln = 1;
  average_adapter.process(interrupted_float, ros::Time(1));
  gnss_comm::GnssPVTSolnMsg shifted = fixedMessage(186158.0);
  shifted.longitude += 1e-7;
  check(!average_adapter.process(shifted, ros::Time(1)).status.origin_initialized,
        "average_fixed must discard a sequence interrupted by RTK Float");
  shifted.time.tow = 186158.1;
  const GnssAdapterResult averaged = average_adapter.process(shifted, ros::Time(1));
  check(averaged.status.origin_initialized && averaged.status.accepted,
        "average_fixed must initialize from consecutive confirmed ECEF samples");
}
} // namespace

int main()
{
  try
  {
    testTimeStateEnuAndCovariance();
    testInvalidInputs();
    testTimeOrderingAndMissingOrigin();
    testLossRecoveryAndAverageOrigin();
  }
  catch (const std::exception &error)
  {
    std::cerr << "gnss_adapter_self_test: FAIL: " << error.what() << std::endl;
    return 1;
  }
  std::cout << "gnss_adapter_self_test: PASS" << std::endl;
  return 0;
}
