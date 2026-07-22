/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#include "gnss_fault_injector.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
constexpr uint64_t kWindowStartNs = 1763437340000000000ULL;
constexpr uint64_t kWindowEndNs = 1763437341000000000ULL;

void check(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

gnss_comm::GnssPVTSolnMsg fixedMessage(double tow)
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

void checkPayloadPreserved(const gnss_comm::GnssPVTSolnMsg &before,
                           const gnss_comm::GnssPVTSolnMsg &after)
{
  check(before.time.week == after.time.week && before.time.tow == after.time.tow,
        "fault injection changed GNSS message time");
  check(before.num_sv == after.num_sv && before.latitude == after.latitude &&
            before.longitude == after.longitude &&
            before.altitude == after.altitude &&
            before.height_msl == after.height_msl && before.h_acc == after.h_acc &&
            before.v_acc == after.v_acc && before.p_dop == after.p_dop &&
            before.vel_n == after.vel_n && before.vel_e == after.vel_e &&
            before.vel_d == after.vel_d && before.vel_acc == after.vel_acc,
        "fault injection changed a non-quality PVT field");
}

void testTimeConversionAndModeParsing()
{
  uint64_t stamp_ns = 0;
  check(GnssFaultInjectorCore::gpsToUtcStampNs(2393, 186157.8, stamp_ns),
        "known GNSS time was rejected");
  const int64_t known_error_ns = static_cast<int64_t>(stamp_ns) -
                                 static_cast<int64_t>(1763437339800000000ULL);
  check(known_error_ns >= -1000 && known_error_ns <= 1000,
        "GPST-to-UTC conversion differs from the adapter");
  check(!GnssFaultInjectorCore::gpsToUtcStampNs(0, 186157.8, stamp_ns),
        "GPS week zero must be invalid");
  check(!GnssFaultInjectorCore::gpsToUtcStampNs(
            2393, std::numeric_limits<double>::quiet_NaN(), stamp_ns),
        "NaN TOW must be invalid");

  GnssFaultMode mode;
  check(parseGnssFaultMode("passthrough", mode) &&
            mode == GnssFaultMode::PASSTHROUGH,
        "passthrough mode parse failed");
  check(parseGnssFaultMode("drop", mode) && mode == GnssFaultMode::DROP,
        "drop mode parse failed");
  check(parseGnssFaultMode("float", mode) && mode == GnssFaultMode::FLOAT,
        "float mode parse failed");
  check(parseGnssFaultMode("invalid", mode) && mode == GnssFaultMode::INVALID,
        "invalid mode parse failed");
  check(!parseGnssFaultMode("DROP", mode), "unknown mode must not be guessed");
}

void testDropHalfOpenWindowAndRecovery()
{
  GnssFaultInjectorCore core(GnssFaultMode::DROP, kWindowStartNs, kWindowEndNs);
  const GnssFaultDecision before = core.process(fixedMessage(186157.9));
  const GnssFaultDecision start = core.process(fixedMessage(186158.0));
  const GnssFaultDecision inside = core.process(fixedMessage(186158.9));
  const GnssFaultDecision end = core.process(fixedMessage(186159.0));

  check(before.publish && !before.in_fault_window,
        "message before drop window must pass");
  check(start.stamp_ns == kWindowStartNs && !start.publish &&
            start.in_fault_window && start.fault_started_now,
        "start boundary must belong to [start,end)");
  check(!inside.publish && inside.in_fault_window && !inside.fault_started_now,
        "interior drop message handling failed");
  check(end.stamp_ns == kWindowEndNs && end.publish &&
            !end.in_fault_window && end.recovered_now,
        "end boundary must be the first recovered message");
  check(core.firstFaultStampNs() == kWindowStartNs &&
            core.lastFaultStampNs() == inside.stamp_ns &&
            core.firstRecoveredStampNs() == kWindowEndNs,
        "fault lifecycle timestamps are wrong");
  check(core.counters().received == 4 && core.counters().passed == 2 &&
            core.counters().dropped == 2 && core.conservationDelta() == 0,
        "drop counters do not conserve received=passed+dropped");
}

void testQualityModesPreservePayload()
{
  const gnss_comm::GnssPVTSolnMsg original = fixedMessage(186158.5);

  GnssFaultInjectorCore float_core(GnssFaultMode::FLOAT, kWindowStartNs,
                                   kWindowEndNs);
  const GnssFaultDecision as_float = float_core.process(original);
  check(as_float.publish && as_float.action == GnssFaultAction::MODIFIED_TO_FLOAT &&
            as_float.message.valid_fix && as_float.message.fix_type == 3 &&
            as_float.message.diff_soln && as_float.message.carr_soln == 1,
        "float mode quality fields are wrong");
  checkPayloadPreserved(original, as_float.message);
  check(float_core.counters().received == 1 &&
            float_core.counters().passed == 1 &&
            float_core.counters().modified_to_float == 1 &&
            float_core.conservationDelta() == 0,
        "float counters are wrong");

  GnssFaultInjectorCore invalid_core(GnssFaultMode::INVALID, kWindowStartNs,
                                     kWindowEndNs);
  const GnssFaultDecision as_invalid = invalid_core.process(original);
  check(as_invalid.publish &&
            as_invalid.action == GnssFaultAction::MODIFIED_TO_INVALID &&
            !as_invalid.message.valid_fix && as_invalid.message.fix_type == 0 &&
            !as_invalid.message.diff_soln && as_invalid.message.carr_soln == 0,
        "invalid mode quality fields are wrong");
  checkPayloadPreserved(original, as_invalid.message);
  check(invalid_core.counters().received == 1 &&
            invalid_core.counters().passed == 1 &&
            invalid_core.counters().modified_to_invalid == 1 &&
            invalid_core.conservationDelta() == 0,
        "invalid counters are wrong");

  GnssFaultInjectorCore passthrough_core(GnssFaultMode::PASSTHROUGH, 0, 0);
  const GnssFaultDecision unchanged = passthrough_core.process(original);
  check(unchanged.publish && !unchanged.in_fault_window &&
            unchanged.message.valid_fix == original.valid_fix &&
            unchanged.message.fix_type == original.fix_type &&
            unchanged.message.diff_soln == original.diff_soln &&
            unchanged.message.carr_soln == original.carr_soln,
        "passthrough mode changed quality fields");
  checkPayloadPreserved(original, unchanged.message);
}

void testInvalidTimePassesUnchanged()
{
  GnssFaultInjectorCore core(GnssFaultMode::DROP, kWindowStartNs, kWindowEndNs);
  gnss_comm::GnssPVTSolnMsg message = fixedMessage(186158.5);
  message.time.week = 0;
  const GnssFaultDecision decision = core.process(message);
  check(decision.publish && !decision.valid_time &&
            !decision.in_fault_window &&
            decision.action == GnssFaultAction::INVALID_TIME_PASSED,
        "invalid GNSS time must pass unchanged without entering the fault window");
  checkPayloadPreserved(message, decision.message);
  check(decision.message.valid_fix == message.valid_fix &&
            decision.message.fix_type == message.fix_type &&
            decision.message.diff_soln == message.diff_soln &&
            decision.message.carr_soln == message.carr_soln,
        "invalid-time passthrough changed quality fields");
  check(core.counters().received == 1 && core.counters().passed == 1 &&
            core.counters().dropped == 0 && core.counters().invalid_time == 1 &&
            core.conservationDelta() == 0,
        "invalid-time counters are wrong");
}

void testInvalidIntervalRejected()
{
  bool threw = false;
  try
  {
    GnssFaultInjectorCore invalid(GnssFaultMode::DROP, kWindowEndNs,
                                  kWindowStartNs);
    (void)invalid;
  }
  catch (const std::invalid_argument &)
  {
    threw = true;
  }
  check(threw, "fault mode must reject start >= end");
}
} // namespace

int main()
{
  try
  {
    testTimeConversionAndModeParsing();
    testDropHalfOpenWindowAndRecovery();
    testQualityModesPreservePayload();
    testInvalidTimePassesUnchanged();
    testInvalidIntervalRejected();
  }
  catch (const std::exception &error)
  {
    std::cerr << "gnss_fault_injector_self_test: FAIL: " << error.what()
              << std::endl;
    return 1;
  }
  std::cout << "gnss_fault_injector_self_test: PASS" << std::endl;
  return 0;
}
