/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#include "gnss_fault_injector.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <gnss_comm/gnss_utility.hpp>
#include <ros/time.h>

namespace
{
constexpr double kGpsWeekSeconds = 604800.0;
constexpr uint32_t kMaxSupportedGpsWeek = 7000;
} // namespace

GnssFaultInjectorCore::GnssFaultInjectorCore(GnssFaultMode mode,
                                             uint64_t start_stamp_ns,
                                             uint64_t end_stamp_ns)
    : mode_(mode), start_stamp_ns_(start_stamp_ns), end_stamp_ns_(end_stamp_ns)
{
  if (mode_ != GnssFaultMode::PASSTHROUGH && start_stamp_ns_ >= end_stamp_ns_)
  {
    throw std::invalid_argument("GNSS fault interval must satisfy start < end");
  }
}

bool GnssFaultInjectorCore::gpsToUtcStampNs(uint32_t week, double tow,
                                            uint64_t &stamp_ns)
{
  stamp_ns = 0;
  if (week == 0 || week > kMaxSupportedGpsWeek || !std::isfinite(tow) ||
      tow < 0.0 || tow >= kGpsWeekSeconds)
  {
    return false;
  }

  const gnss_comm::gtime_t gps_time = gnss_comm::gpst2time(week, tow);
  const double unix_utc = gnss_comm::time2sec(gnss_comm::gpst2utc(gps_time));
  if (!std::isfinite(unix_utc) || unix_utc <= 0.0 ||
      unix_utc > static_cast<double>(std::numeric_limits<uint32_t>::max()))
  {
    return false;
  }

  ros::Time stamp;
  stamp.fromSec(unix_utc);
  if (stamp.isZero()) return false;
  stamp_ns = stamp.toNSec();
  return true;
}

GnssFaultDecision GnssFaultInjectorCore::process(
    const gnss_comm::GnssPVTSolnMsg &message)
{
  ++counters_.received;
  GnssFaultDecision result;
  result.message = message;
  result.valid_time = gpsToUtcStampNs(message.time.week, message.time.tow,
                                      result.stamp_ns);
  if (!result.valid_time)
  {
    // ponytail: never invent callback time for a message-time experiment.
    result.action = GnssFaultAction::INVALID_TIME_PASSED;
    ++counters_.invalid_time;
    ++counters_.passed;
    return result;
  }

  result.in_fault_window = mode_ != GnssFaultMode::PASSTHROUGH &&
                           result.stamp_ns >= start_stamp_ns_ &&
                           result.stamp_ns < end_stamp_ns_;
  if (result.in_fault_window)
  {
    if (!have_first_fault_stamp_)
    {
      have_first_fault_stamp_ = true;
      first_fault_stamp_ns_ = result.stamp_ns;
      result.fault_started_now = true;
    }
    last_fault_stamp_ns_ = result.stamp_ns;

    switch (mode_)
    {
      case GnssFaultMode::DROP:
        result.action = GnssFaultAction::DROPPED;
        result.publish = false;
        ++counters_.dropped;
        return result;
      case GnssFaultMode::FLOAT:
        result.action = GnssFaultAction::MODIFIED_TO_FLOAT;
        result.message.valid_fix = true;
        result.message.fix_type = 3;
        result.message.diff_soln = true;
        result.message.carr_soln = 1;
        ++counters_.modified_to_float;
        break;
      case GnssFaultMode::INVALID:
        result.action = GnssFaultAction::MODIFIED_TO_INVALID;
        result.message.valid_fix = false;
        result.message.fix_type = 0;
        result.message.diff_soln = false;
        result.message.carr_soln = 0;
        ++counters_.modified_to_invalid;
        break;
      case GnssFaultMode::PASSTHROUGH:
        break;
    }
  }
  else if (have_first_fault_stamp_ && !have_first_recovered_stamp_ &&
           result.stamp_ns >= end_stamp_ns_)
  {
    have_first_recovered_stamp_ = true;
    first_recovered_stamp_ns_ = result.stamp_ns;
    result.recovered_now = true;
  }

  ++counters_.passed;
  return result;
}

int64_t GnssFaultInjectorCore::conservationDelta() const
{
  return static_cast<int64_t>(counters_.received) -
         static_cast<int64_t>(counters_.passed) -
         static_cast<int64_t>(counters_.dropped);
}

bool parseGnssFaultMode(const std::string &value, GnssFaultMode &mode)
{
  if (value == "passthrough") mode = GnssFaultMode::PASSTHROUGH;
  else if (value == "drop") mode = GnssFaultMode::DROP;
  else if (value == "float") mode = GnssFaultMode::FLOAT;
  else if (value == "invalid") mode = GnssFaultMode::INVALID;
  else return false;
  return true;
}

const char *gnssFaultModeName(GnssFaultMode mode)
{
  switch (mode)
  {
    case GnssFaultMode::PASSTHROUGH: return "passthrough";
    case GnssFaultMode::DROP: return "drop";
    case GnssFaultMode::FLOAT: return "float";
    case GnssFaultMode::INVALID: return "invalid";
  }
  return "unknown";
}

const char *gnssFaultActionName(GnssFaultAction action)
{
  switch (action)
  {
    case GnssFaultAction::PASSED: return "PASSED";
    case GnssFaultAction::DROPPED: return "DROPPED";
    case GnssFaultAction::MODIFIED_TO_FLOAT: return "MODIFIED_TO_FLOAT";
    case GnssFaultAction::MODIFIED_TO_INVALID: return "MODIFIED_TO_INVALID";
    case GnssFaultAction::INVALID_TIME_PASSED: return "INVALID_TIME_PASSED";
  }
  return "UNKNOWN";
}
