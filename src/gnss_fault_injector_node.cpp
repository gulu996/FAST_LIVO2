/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#include "gnss_fault_injector.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>

#include <ros/ros.h>
#include <std_msgs/String.h>

namespace
{
constexpr int kQueueSize = 100;

std::string stampText(uint64_t stamp_ns)
{
  std::ostringstream stream;
  stream << stamp_ns / 1000000000ULL << "." << std::setfill('0')
         << std::setw(9) << stamp_ns % 1000000000ULL;
  return stream.str();
}

bool secondsToStampNs(double seconds, uint64_t &stamp_ns)
{
  stamp_ns = 0;
  if (!std::isfinite(seconds) || seconds < 0.0 ||
      seconds > static_cast<double>(std::numeric_limits<uint32_t>::max()))
  {
    return false;
  }
  ros::Time stamp;
  stamp.fromSec(seconds);
  stamp_ns = stamp.toNSec();
  return true;
}

bool openCsv(const std::string &path, std::ofstream &stream)
{
  if (path.empty()) return false;
  const std::filesystem::path file(path);
  const std::filesystem::path parent = file.parent_path();
  if (!parent.empty())
  {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error)
    {
      ROS_ERROR("[GNSS_FAULT] Cannot create %s: %s", parent.c_str(),
                error.message().c_str());
      return false;
    }
  }
  stream.open(file, std::ios::out | std::ios::trunc);
  return static_cast<bool>(stream);
}
} // namespace

class GnssFaultInjectorNode
{
public:
  GnssFaultInjectorNode() : private_nh_("~") {}

  ~GnssFaultInjectorNode() { finalize(); }

  bool initialize()
  {
    private_nh_.param("enable", enabled_, true);
    private_nh_.param<std::string>("input_topic", input_topic_,
                                   "/ublox_driver/receiver_pvt");
    private_nh_.param<std::string>("output_topic", output_topic_,
                                   "/gnss_fault/receiver_pvt");
    private_nh_.param<std::string>("mode", requested_mode_name_, "passthrough");
    private_nh_.param("start_stamp", start_stamp_seconds_, 0.0);
    private_nh_.param("end_stamp", end_stamp_seconds_, 0.0);
    private_nh_.param("use_message_time", use_message_time_, true);
    private_nh_.param("log_every_faulted_message", log_every_faulted_message_, false);
    private_nh_.param<std::string>("status_topic", status_topic_,
                                   "/gnss_fault/status");
    private_nh_.param<std::string>("event_log_file", event_log_file_,
                                   "/tmp/fast_livo_rtk/gnss_fault_events.csv");
    private_nh_.param<std::string>("status_csv_file", status_csv_file_,
                                   "/tmp/fast_livo_rtk/gnss_fault_status.csv");

    GnssFaultMode requested_mode;
    if (!parseGnssFaultMode(requested_mode_name_, requested_mode))
    {
      ROS_ERROR("[GNSS_FAULT] Unsupported mode: %s", requested_mode_name_.c_str());
      return false;
    }
    if (!use_message_time_)
    {
      ROS_ERROR("[GNSS_FAULT] use_message_time=false is forbidden for deterministic replay.");
      return false;
    }

    uint64_t start_stamp_ns = 0;
    uint64_t end_stamp_ns = 0;
    if (!secondsToStampNs(start_stamp_seconds_, start_stamp_ns) ||
        !secondsToStampNs(end_stamp_seconds_, end_stamp_ns))
    {
      ROS_ERROR("[GNSS_FAULT] start_stamp/end_stamp must be finite ROS timestamps.");
      return false;
    }

    const GnssFaultMode effective_mode = enabled_ ? requested_mode
                                                   : GnssFaultMode::PASSTHROUGH;
    try
    {
      core_ = std::make_unique<GnssFaultInjectorCore>(effective_mode,
                                                       start_stamp_ns,
                                                       end_stamp_ns);
    }
    catch (const std::exception &error)
    {
      ROS_ERROR("[GNSS_FAULT] Invalid configuration: %s", error.what());
      return false;
    }

    if (node_nh_.resolveName(input_topic_) == node_nh_.resolveName(output_topic_))
    {
      ROS_ERROR("[GNSS_FAULT] input_topic and output_topic must differ.");
      return false;
    }
    if (event_log_file_ == status_csv_file_)
    {
      ROS_ERROR("[GNSS_FAULT] event_log_file and status_csv_file must differ.");
      return false;
    }
    if (!openCsv(event_log_file_, event_stream_) ||
        !openCsv(status_csv_file_, status_stream_))
    {
      ROS_ERROR("[GNSS_FAULT] Failed to open deterministic CSV outputs.");
      return false;
    }

    const std::string common_header =
        "stamp_ns,stamp,event,action,valid_time,in_fault_window,enabled,mode,"
        "start_stamp_ns,end_stamp_ns,received,passed,dropped,modified_to_float,"
        "modified_to_invalid,invalid_time,first_fault_stamp_ns,"
        "last_fault_stamp_ns,first_recovered_stamp_ns,conservation_delta";
    event_stream_ << "event_index," << common_header << "\n";
    status_stream_ << "sequence," << common_header << "\n";
    writeEvent("START", 0, "NONE", false, false);

    output_publisher_ = node_nh_.advertise<gnss_comm::GnssPVTSolnMsg>(
        output_topic_, kQueueSize);
    status_publisher_ = node_nh_.advertise<std_msgs::String>(status_topic_, 1, true);
    input_subscriber_ = node_nh_.subscribe(input_topic_, kQueueSize,
                                           &GnssFaultInjectorNode::pvtCallback,
                                           this);
    publishStatus("START", 0, "NONE", false, false);

    ROS_INFO("[GNSS_FAULT] enabled=%s mode=%s input=%s output=%s window=[%s,%s)",
             enabled_ ? "true" : "false", gnssFaultModeName(core_->mode()),
             input_topic_.c_str(), output_topic_.c_str(),
             stampText(core_->startStampNs()).c_str(),
             stampText(core_->endStampNs()).c_str());
    return true;
  }

  void finalize()
  {
    if (!core_ || finalized_) return;
    finalized_ = true;
    const int64_t delta = core_->conservationDelta();
    writeEvent(delta == 0 ? "SHUTDOWN" : "CONSERVATION_ERROR",
               last_stamp_ns_, "NONE", last_valid_time_, false);
    if (status_stream_)
    {
      writeStatusRow("SHUTDOWN", last_stamp_ns_, "NONE", last_valid_time_, false);
    }
    event_stream_.flush();
    status_stream_.flush();
    if (delta != 0)
    {
      ROS_ERROR("[GNSS_FAULT] Counter conservation failed: delta=%ld",
                static_cast<long>(delta));
    }
    const GnssFaultCounters &counts = core_->counters();
    ROS_INFO("[GNSS_FAULT_SUMMARY] received=%lu passed=%lu dropped=%lu float=%lu "
             "invalid=%lu invalid_time=%lu conservation_delta=%ld",
             static_cast<unsigned long>(counts.received),
             static_cast<unsigned long>(counts.passed),
             static_cast<unsigned long>(counts.dropped),
             static_cast<unsigned long>(counts.modified_to_float),
             static_cast<unsigned long>(counts.modified_to_invalid),
             static_cast<unsigned long>(counts.invalid_time),
             static_cast<long>(delta));
  }

private:
  std::string optionalStamp(bool present, uint64_t stamp_ns) const
  {
    return present ? std::to_string(stamp_ns) : std::string();
  }

  std::string commonCsv(uint64_t stamp_ns, const std::string &event,
                        const std::string &action, bool valid_time,
                        bool in_fault_window) const
  {
    const GnssFaultCounters &counts = core_->counters();
    std::ostringstream row;
    row << stamp_ns << "," << stampText(stamp_ns) << "," << event << ","
        << action << "," << (valid_time ? 1 : 0) << ","
        << (in_fault_window ? 1 : 0) << "," << (enabled_ ? 1 : 0) << ","
        << gnssFaultModeName(core_->mode()) << "," << core_->startStampNs()
        << "," << core_->endStampNs() << "," << counts.received << ","
        << counts.passed << "," << counts.dropped << ","
        << counts.modified_to_float << "," << counts.modified_to_invalid
        << "," << counts.invalid_time << ","
        << optionalStamp(core_->haveFirstFaultStamp(), core_->firstFaultStampNs())
        << ","
        << optionalStamp(core_->haveFirstFaultStamp(), core_->lastFaultStampNs())
        << ","
        << optionalStamp(core_->haveFirstRecoveredStamp(),
                         core_->firstRecoveredStampNs())
        << "," << core_->conservationDelta();
    return row.str();
  }

  void writeEvent(const std::string &event, uint64_t stamp_ns,
                  const std::string &action, bool valid_time,
                  bool in_fault_window)
  {
    if (!event_stream_) return;
    event_stream_ << event_index_++ << ","
                  << commonCsv(stamp_ns, event, action, valid_time,
                               in_fault_window)
                  << "\n";
    event_stream_.flush();
  }

  void writeStatusRow(const std::string &event, uint64_t stamp_ns,
                      const std::string &action, bool valid_time,
                      bool in_fault_window)
  {
    status_stream_ << sequence_++ << ","
                   << commonCsv(stamp_ns, event, action, valid_time,
                                in_fault_window)
                   << "\n";
    status_stream_.flush();
  }

  void publishStatus(const std::string &event, uint64_t stamp_ns,
                     const std::string &action, bool valid_time,
                     bool in_fault_window)
  {
    const GnssFaultCounters &counts = core_->counters();
    std_msgs::String message;
    std::ostringstream text;
    text << "event=" << event << " action=" << action
         << " stamp_ns=" << stamp_ns
         << " valid_time=" << (valid_time ? 1 : 0)
         << " in_fault_window=" << (in_fault_window ? 1 : 0)
         << " enabled=" << (enabled_ ? 1 : 0)
         << " mode=" << gnssFaultModeName(core_->mode())
         << " received=" << counts.received << " passed=" << counts.passed
         << " dropped=" << counts.dropped
         << " modified_to_float=" << counts.modified_to_float
         << " modified_to_invalid=" << counts.modified_to_invalid
         << " invalid_time=" << counts.invalid_time
         << " first_fault_stamp_ns="
         << optionalStamp(core_->haveFirstFaultStamp(), core_->firstFaultStampNs())
         << " last_fault_stamp_ns="
         << optionalStamp(core_->haveFirstFaultStamp(), core_->lastFaultStampNs())
         << " first_recovered_stamp_ns="
         << optionalStamp(core_->haveFirstRecoveredStamp(),
                          core_->firstRecoveredStampNs())
         << " conservation_delta=" << core_->conservationDelta();
    message.data = text.str();
    status_publisher_.publish(message);
  }

  void pvtCallback(const gnss_comm::GnssPVTSolnMsgConstPtr &message)
  {
    const GnssFaultDecision decision = core_->process(*message);
    last_stamp_ns_ = decision.stamp_ns;
    last_valid_time_ = decision.valid_time;

    std::string event = "MESSAGE";
    if (!decision.valid_time) event = "INVALID_MESSAGE_TIME";
    else if (decision.fault_started_now) event = "FAULT_BEGIN";
    else if (decision.recovered_now) event = "RECOVERED";
    else if (decision.in_fault_window) event = "FAULTED";
    const std::string action = gnssFaultActionName(decision.action);

    if (decision.fault_started_now || decision.recovered_now ||
        !decision.valid_time ||
        (log_every_faulted_message_ && decision.in_fault_window))
    {
      writeEvent(event, decision.stamp_ns, action, decision.valid_time,
                 decision.in_fault_window);
    }
    writeStatusRow(event, decision.stamp_ns, action, decision.valid_time,
                   decision.in_fault_window);
    publishStatus(event, decision.stamp_ns, action, decision.valid_time,
                  decision.in_fault_window);

    if (decision.publish) output_publisher_.publish(decision.message);
    if (decision.fault_started_now)
    {
      ROS_WARN("[GNSS_FAULT_BEGIN] mode=%s stamp=%s",
               gnssFaultModeName(core_->mode()),
               stampText(decision.stamp_ns).c_str());
    }
    if (decision.recovered_now)
    {
      ROS_INFO("[GNSS_FAULT_RECOVERED] first_stamp=%s",
               stampText(decision.stamp_ns).c_str());
    }
    if (!decision.valid_time)
    {
      ROS_WARN("[GNSS_FAULT] Invalid week/tow passed through unchanged.");
    }
    else if (log_every_faulted_message_ && decision.in_fault_window)
    {
      ROS_INFO("[GNSS_FAULT_MESSAGE] action=%s stamp=%s", action.c_str(),
               stampText(decision.stamp_ns).c_str());
    }
  }

  ros::NodeHandle node_nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber input_subscriber_;
  ros::Publisher output_publisher_;
  ros::Publisher status_publisher_;
  std::unique_ptr<GnssFaultInjectorCore> core_;
  std::ofstream event_stream_;
  std::ofstream status_stream_;

  bool enabled_ = true;
  bool use_message_time_ = true;
  bool log_every_faulted_message_ = false;
  bool finalized_ = false;
  bool last_valid_time_ = false;
  double start_stamp_seconds_ = 0.0;
  double end_stamp_seconds_ = 0.0;
  uint64_t event_index_ = 0;
  uint64_t sequence_ = 0;
  uint64_t last_stamp_ns_ = 0;
  std::string input_topic_;
  std::string output_topic_;
  std::string requested_mode_name_;
  std::string status_topic_;
  std::string event_log_file_;
  std::string status_csv_file_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gnss_fault_injector_node");
  GnssFaultInjectorNode node;
  if (!node.initialize()) return 1;
  ros::spin();
  node.finalize();
  return 0;
}
