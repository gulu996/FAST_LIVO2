/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#ifndef GNSS_FAULT_INJECTOR_H
#define GNSS_FAULT_INJECTOR_H

#include <cstdint>
#include <string>

#include <gnss_comm/GnssPVTSolnMsg.h>

enum class GnssFaultMode : uint8_t
{
  PASSTHROUGH = 0,
  DROP,
  FLOAT,
  INVALID
};

enum class GnssFaultAction : uint8_t
{
  PASSED = 0,
  DROPPED,
  MODIFIED_TO_FLOAT,
  MODIFIED_TO_INVALID,
  INVALID_TIME_PASSED
};

struct GnssFaultCounters
{
  uint64_t received = 0;
  uint64_t passed = 0;
  uint64_t dropped = 0;
  uint64_t modified_to_float = 0;
  uint64_t modified_to_invalid = 0;
  uint64_t invalid_time = 0;
};

struct GnssFaultDecision
{
  gnss_comm::GnssPVTSolnMsg message;
  GnssFaultAction action = GnssFaultAction::PASSED;
  bool publish = true;
  bool valid_time = false;
  bool in_fault_window = false;
  bool fault_started_now = false;
  bool recovered_now = false;
  uint64_t stamp_ns = 0;
};

class GnssFaultInjectorCore
{
public:
  GnssFaultInjectorCore(GnssFaultMode mode, uint64_t start_stamp_ns,
                        uint64_t end_stamp_ns);

  GnssFaultDecision process(const gnss_comm::GnssPVTSolnMsg &message);

  const GnssFaultCounters &counters() const { return counters_; }
  GnssFaultMode mode() const { return mode_; }
  uint64_t startStampNs() const { return start_stamp_ns_; }
  uint64_t endStampNs() const { return end_stamp_ns_; }
  bool haveFirstFaultStamp() const { return have_first_fault_stamp_; }
  bool haveFirstRecoveredStamp() const { return have_first_recovered_stamp_; }
  uint64_t firstFaultStampNs() const { return first_fault_stamp_ns_; }
  uint64_t lastFaultStampNs() const { return last_fault_stamp_ns_; }
  uint64_t firstRecoveredStampNs() const { return first_recovered_stamp_ns_; }
  int64_t conservationDelta() const;

  static bool gpsToUtcStampNs(uint32_t week, double tow, uint64_t &stamp_ns);

private:
  GnssFaultMode mode_ = GnssFaultMode::PASSTHROUGH;
  uint64_t start_stamp_ns_ = 0;
  uint64_t end_stamp_ns_ = 0;
  GnssFaultCounters counters_;
  bool have_first_fault_stamp_ = false;
  bool have_first_recovered_stamp_ = false;
  uint64_t first_fault_stamp_ns_ = 0;
  uint64_t last_fault_stamp_ns_ = 0;
  uint64_t first_recovered_stamp_ns_ = 0;
};

bool parseGnssFaultMode(const std::string &value, GnssFaultMode &mode);
const char *gnssFaultModeName(GnssFaultMode mode);
const char *gnssFaultActionName(GnssFaultAction action);

#endif // GNSS_FAULT_INJECTOR_H
