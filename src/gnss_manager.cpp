/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#include "gnss_manager.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <xmlrpcpp/XmlRpcValue.h>

namespace
{
constexpr double kDegToRad = M_PI / 180.0;
constexpr double kRadToDeg = 180.0 / M_PI;
constexpr double kWgs84A = 6378137.0;
constexpr double kWgs84F = 1.0 / 298.257223563;
constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);

std::string toLower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string trimLine(const std::string &line)
{
  const size_t first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const size_t last = line.find_last_not_of(" \t\r\n");
  return line.substr(first, last - first + 1);
}

bool xmlRpcToInt(const XmlRpc::XmlRpcValue &value, int &out)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    out = static_cast<int>(value);
    return true;
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
  {
    const double d = static_cast<double>(value);
    out = static_cast<int>(std::llround(d));
    return std::isfinite(d) && std::fabs(d - static_cast<double>(out)) < 1e-6;
  }
  return false;
}

bool xmlRpcToDouble(const XmlRpc::XmlRpcValue &value, double &out)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
  {
    out = static_cast<double>(value);
    return true;
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    out = static_cast<int>(value);
    return true;
  }
  return false;
}

bool loadIntSetParam(ros::NodeHandle &nh, const std::string &name, std::set<int> &out)
{
  XmlRpc::XmlRpcValue values;
  if (!nh.getParam(name, values)) return false;
  if (values.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;

  std::set<int> parsed;
  for (int i = 0; i < values.size(); ++i)
  {
    int value = 0;
    if (!xmlRpcToInt(values[i], value)) return false;
    parsed.insert(value);
  }
  out = parsed;
  return true;
}

bool loadVec3Param(ros::NodeHandle &nh, const std::string &name, V3D &out)
{
  XmlRpc::XmlRpcValue values;
  if (!nh.getParam(name, values)) return false;
  if (values.getType() != XmlRpc::XmlRpcValue::TypeArray || values.size() != 3) return false;

  double x = 0.0, y = 0.0, z = 0.0;
  if (!xmlRpcToDouble(values[0], x) ||
      !xmlRpcToDouble(values[1], y) ||
      !xmlRpcToDouble(values[2], z))
  {
    return false;
  }
  out << x, y, z;
  return true;
}

bool baudrateToTermios(int baudrate, speed_t &speed)
{
  switch (baudrate)
  {
    case 9600: speed = B9600; return true;
    case 19200: speed = B19200; return true;
    case 38400: speed = B38400; return true;
    case 57600: speed = B57600; return true;
    case 115200: speed = B115200; return true;
#ifdef B230400
    case 230400: speed = B230400; return true;
#endif
#ifdef B460800
    case 460800: speed = B460800; return true;
#endif
#ifdef B500000
    case 500000: speed = B500000; return true;
#endif
#ifdef B576000
    case 576000: speed = B576000; return true;
#endif
#ifdef B921600
    case 921600: speed = B921600; return true;
#endif
#ifdef B1000000
    case 1000000: speed = B1000000; return true;
#endif
#ifdef B1152000
    case 1152000: speed = B1152000; return true;
#endif
#ifdef B1500000
    case 1500000: speed = B1500000; return true;
#endif
#ifdef B2000000
    case 2000000: speed = B2000000; return true;
#endif
#ifdef B2500000
    case 2500000: speed = B2500000; return true;
#endif
#ifdef B3000000
    case 3000000: speed = B3000000; return true;
#endif
#ifdef B3500000
    case 3500000: speed = B3500000; return true;
#endif
#ifdef B4000000
    case 4000000: speed = B4000000; return true;
#endif
    default: return false;
  }
}

bool finiteVec3(const V3D &v)
{
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool startsWith(const std::string &value, const std::string &prefix)
{
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::vector<std::string> splitPreserveEmpty(const std::string &value, char delimiter)
{
  std::vector<std::string> fields;
  size_t start = 0;
  while (start <= value.size())
  {
    const size_t pos = value.find(delimiter, start);
    if (pos == std::string::npos)
    {
      fields.push_back(value.substr(start));
      break;
    }
    fields.push_back(value.substr(start, pos - start));
    start = pos + 1;
  }
  return fields;
}

std::string upperString(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string trimField(const std::string &field)
{
  return trimLine(field);
}

bool parseDoubleField(const std::string &field, double &value)
{
  const std::string trimmed = trimField(field);
  if (trimmed.empty()) return false;
  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(trimmed.c_str(), &end);
  if (end == trimmed.c_str() || errno == ERANGE || !std::isfinite(parsed)) return false;
  while (*end != '\0')
  {
    if (!std::isspace(static_cast<unsigned char>(*end))) return false;
    ++end;
  }
  value = parsed;
  return true;
}

bool parseIntField(const std::string &field, int &value)
{
  double parsed = 0.0;
  if (!parseDoubleField(field, parsed)) return false;
  const double rounded = std::round(parsed);
  if (std::fabs(parsed - rounded) > 1e-6) return false;
  value = static_cast<int>(rounded);
  return true;
}

bool parseHexByte(const std::string &hex, uint8_t &value)
{
  if (hex.size() < 2) return false;
  auto hexValue = [](char c, int &out) {
    if (c >= '0' && c <= '9')
    {
      out = c - '0';
      return true;
    }
    if (c >= 'a' && c <= 'f')
    {
      out = 10 + c - 'a';
      return true;
    }
    if (c >= 'A' && c <= 'F')
    {
      out = 10 + c - 'A';
      return true;
    }
    return false;
  };
  int hi = 0, lo = 0;
  if (!hexValue(hex[0], hi) || !hexValue(hex[1], lo)) return false;
  value = static_cast<uint8_t>((hi << 4) | lo);
  return true;
}

bool parseNmeaFields(const std::string &line, std::vector<std::string> &fields, bool &checksum_valid)
{
  checksum_valid = false;
  if (line.empty() || line.front() != '$') return false;
  const size_t star = line.find('*');
  if (star == std::string::npos || star + 2 >= line.size()) return false;

  uint8_t expected = 0;
  if (!parseHexByte(line.substr(star + 1, 2), expected)) return false;

  uint8_t checksum = 0;
  for (size_t i = 1; i < star; ++i)
  {
    checksum ^= static_cast<uint8_t>(line[i]);
  }
  checksum_valid = checksum == expected;
  const std::string body = line.substr(1, star - 1);
  fields = splitPreserveEmpty(body, ',');
  if (!fields.empty()) fields[0] = "$" + fields[0];
  return true;
}

bool parseNmeaLatLon(const std::string &value_field, const std::string &hemisphere_field,
                     bool latitude, double &degrees)
{
  double raw = 0.0;
  if (!parseDoubleField(value_field, raw)) return false;
  const std::string hemisphere = upperString(trimField(hemisphere_field));
  if (hemisphere.empty()) return false;
  const int deg_digits = latitude ? 2 : 3;
  const double scale = std::pow(10.0, deg_digits);
  const double deg_part = std::floor(raw / 100.0);
  const double min_part = raw - deg_part * 100.0;
  if (deg_part >= scale || min_part < 0.0 || min_part >= 60.0) return false;
  degrees = deg_part + min_part / 60.0;
  if (hemisphere == "S" || hemisphere == "W") degrees = -degrees;
  if ((latitude && hemisphere != "N" && hemisphere != "S") ||
      (!latitude && hemisphere != "E" && hemisphere != "W"))
  {
    return false;
  }
  return true;
}

bool parseKsxtUtc(const std::string &field, double &unix_stamp)
{
  const std::string trimmed = trimField(field);
  if (trimmed.size() < 14) return false;
  double raw = 0.0;
  if (!parseDoubleField(trimmed, raw) || raw <= 0.0) return false;

  const std::string whole = trimmed.substr(0, 14);
  for (char c : whole)
  {
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  }
  std::tm tm{};
  tm.tm_year = std::stoi(whole.substr(0, 4)) - 1900;
  tm.tm_mon = std::stoi(whole.substr(4, 2)) - 1;
  tm.tm_mday = std::stoi(whole.substr(6, 2));
  tm.tm_hour = std::stoi(whole.substr(8, 2));
  tm.tm_min = std::stoi(whole.substr(10, 2));
  tm.tm_sec = std::stoi(whole.substr(12, 2));
  if (tm.tm_year < 100 || tm.tm_mon < 0 || tm.tm_mon > 11 ||
      tm.tm_mday < 1 || tm.tm_mday > 31 ||
      tm.tm_hour < 0 || tm.tm_hour > 23 ||
      tm.tm_min < 0 || tm.tm_min > 59 ||
      tm.tm_sec < 0 || tm.tm_sec > 60)
  {
    return false;
  }
  const time_t seconds = timegm(&tm);
  if (seconds <= 0) return false;
  const size_t dot = trimmed.find('.');
  double frac = 0.0;
  if (dot != std::string::npos)
  {
    frac = std::strtod(("0" + trimmed.substr(dot)).c_str(), nullptr);
  }
  unix_stamp = static_cast<double>(seconds) + frac;
  return std::isfinite(unix_stamp);
}

const char *solutionTypeName(GnssSolutionType type)
{
  switch (type)
  {
    case GnssSolutionType::INVALID: return "INVALID";
    case GnssSolutionType::SINGLE: return "SINGLE";
    case GnssSolutionType::DIFFERENTIAL: return "DIFFERENTIAL";
    case GnssSolutionType::RTK_FLOAT: return "RTK_FLOAT";
    case GnssSolutionType::RTK_FIXED: return "RTK_FIXED";
    case GnssSolutionType::MANUAL_FIXED: return "MANUAL_FIXED";
    case GnssSolutionType::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

bool validLatLon(double latitude_deg, double longitude_deg)
{
  return std::isfinite(latitude_deg) && std::isfinite(longitude_deg) &&
         latitude_deg >= -90.0 && latitude_deg <= 90.0 &&
         longitude_deg >= -180.0 && longitude_deg <= 180.0;
}

bool zeroLatLon(double latitude_deg, double longitude_deg)
{
  return std::fabs(latitude_deg) < 1e-12 && std::fabs(longitude_deg) < 1e-12;
}

bool findJsonNumber(const std::string &json, const std::string &key, double &value)
{
  const std::string token = "\"" + key + "\"";
  size_t pos = json.find(token);
  if (pos == std::string::npos) return false;
  pos = json.find(':', pos + token.size());
  if (pos == std::string::npos) return false;
  ++pos;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  if (pos >= json.size()) return false;

  const char *start = json.c_str() + pos;
  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(start, &end);
  if (end == start || errno == ERANGE || !std::isfinite(parsed)) return false;
  value = parsed;
  return true;
}

bool findJsonInt(const std::string &json, const std::string &key, int &value)
{
  double d = 0.0;
  if (!findJsonNumber(json, key, d)) return false;
  const double rounded = std::round(d);
  if (std::fabs(d - rounded) > 1e-6) return false;
  value = static_cast<int>(rounded);
  return true;
}

bool extractJsonObject(const std::string &line, size_t prefix_pos, std::string &json, size_t &next_pos)
{
  const size_t brace_pos = line.find('{', prefix_pos);
  if (brace_pos == std::string::npos) return false;

  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = brace_pos; i < line.size(); ++i)
  {
    const char c = line[i];
    if (in_string)
    {
      if (escape)
      {
        escape = false;
      }
      else if (c == '\\')
      {
        escape = true;
      }
      else if (c == '"')
      {
        in_string = false;
      }
      continue;
    }

    if (c == '"')
    {
      in_string = true;
    }
    else if (c == '{')
    {
      depth++;
    }
    else if (c == '}')
    {
      depth--;
      if (depth == 0)
      {
        json = line.substr(brace_pos, i - brace_pos + 1);
        next_pos = i + 1;
        return true;
      }
    }
  }
  return false;
}

bool parseImuGnssJsonObject(const std::string &json,
                            const std::string &raw_line,
                            double stamp,
                            GnssMeasurement &measurement)
{
  measurement = GnssMeasurement();
  measurement.stamp = stamp;
  measurement.receive_stamp = stamp;
  measurement.device_stamp = 0.0;
  measurement.device_time_valid = false;
  measurement.raw_line = raw_line;
  measurement.source_message = "LEGACY_IMUGNSS_JSON";
  measurement.checksum_valid = true;

  bool ok = true;
  ok = findJsonInt(json, "seq", measurement.seq) && ok;
  ok = findJsonNumber(json, "lat", measurement.latitude_deg) && ok;
  ok = findJsonNumber(json, "lon", measurement.longitude_deg) && ok;
  ok = findJsonNumber(json, "alt", measurement.altitude_m) && ok;
  ok = findJsonNumber(json, "roll", measurement.roll_deg) && ok;
  ok = findJsonNumber(json, "pitch", measurement.pitch_deg) && ok;
  ok = findJsonNumber(json, "yaw", measurement.yaw_deg) && ok;
  ok = findJsonNumber(json, "ve", measurement.velocity_east) && ok;
  ok = findJsonNumber(json, "vn", measurement.velocity_north) && ok;
  ok = findJsonNumber(json, "vu", measurement.velocity_up) && ok;
  ok = findJsonInt(json, "state", measurement.state) && ok;
  measurement.raw_position_quality = measurement.state;

  if (!ok)
  {
    measurement.valid = false;
    measurement.reject_reason = "reject_invalid";
    return false;
  }

  const bool all_finite =
      std::isfinite(measurement.latitude_deg) &&
      std::isfinite(measurement.longitude_deg) &&
      std::isfinite(measurement.altitude_m) &&
      std::isfinite(measurement.roll_deg) &&
      std::isfinite(measurement.pitch_deg) &&
      std::isfinite(measurement.yaw_deg) &&
      std::isfinite(measurement.velocity_east) &&
      std::isfinite(measurement.velocity_north) &&
      std::isfinite(measurement.velocity_up);
  if (!all_finite ||
      measurement.latitude_deg < -90.0 || measurement.latitude_deg > 90.0 ||
      measurement.longitude_deg < -180.0 || measurement.longitude_deg > 180.0)
  {
    measurement.valid = false;
    measurement.reject_reason = "reject_invalid";
    return false;
  }
  if (std::fabs(measurement.latitude_deg) < 1e-12 &&
      std::fabs(measurement.longitude_deg) < 1e-12)
  {
    measurement.valid = false;
    measurement.reject_reason = "reject_zero_position";
    return false;
  }

  measurement.valid = true;
  measurement.solution_type = measurement.state == 4 ? GnssSolutionType::RTK_FIXED :
                              (measurement.state == 5 ? GnssSolutionType::RTK_FLOAT :
                               (measurement.state == 0 ? GnssSolutionType::INVALID : GnssSolutionType::UNKNOWN));
  measurement.reject_reason = "ok";
  return true;
}

double xyDistance(const V3D &a, const V3D &b)
{
  return std::hypot(a.x() - b.x(), a.y() - b.y());
}

double sampleMotionExtent(const std::vector<GnssFrameAlignSample> &samples)
{
  if (samples.size() < 2) return 0.0;
  double max_motion = 0.0;
  for (size_t i = 1; i < samples.size(); ++i)
  {
    max_motion = std::max(max_motion, xyDistance(samples.front().enu_position, samples[i].enu_position));
  }
  return max_motion;
}

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}
} // namespace

V3D geodeticToEcef(double latitude_deg, double longitude_deg, double ellipsoid_height_m)
{
  const double lat = latitude_deg * kDegToRad;
  const double lon = longitude_deg * kDegToRad;
  const double sin_lat = std::sin(lat);
  const double cos_lat = std::cos(lat);
  const double sin_lon = std::sin(lon);
  const double cos_lon = std::cos(lon);
  const double n = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sin_lat * sin_lat);

  V3D ecef;
  ecef.x() = (n + ellipsoid_height_m) * cos_lat * cos_lon;
  ecef.y() = (n + ellipsoid_height_m) * cos_lat * sin_lon;
  ecef.z() = (n * (1.0 - kWgs84E2) + ellipsoid_height_m) * sin_lat;
  return ecef;
}

V3D ecefToEnu(const V3D &ecef, const V3D &origin_ecef,
              double origin_latitude_deg, double origin_longitude_deg)
{
  const double lat = origin_latitude_deg * kDegToRad;
  const double lon = origin_longitude_deg * kDegToRad;
  const double sin_lat = std::sin(lat);
  const double cos_lat = std::cos(lat);
  const double sin_lon = std::sin(lon);
  const double cos_lon = std::cos(lon);
  const V3D d = ecef - origin_ecef;

  V3D enu;
  enu.x() = -sin_lon * d.x() + cos_lon * d.y();
  enu.y() = -sin_lat * cos_lon * d.x() - sin_lat * sin_lon * d.y() + cos_lat * d.z();
  enu.z() = cos_lat * cos_lon * d.x() + cos_lat * sin_lon * d.y() + sin_lat * d.z();
  return enu;
}

std::vector<GnssMeasurement> parseImuGnssJsonLine(const std::string &line, double stamp)
{
  std::vector<GnssMeasurement> measurements;
  static const std::string kPrefix = "@IMUGNSS:";

  size_t pos = line.find(kPrefix);
  while (pos != std::string::npos)
  {
    std::string json;
    size_t next_pos = pos + kPrefix.size();
    if (!extractJsonObject(line, pos + kPrefix.size(), json, next_pos))
    {
      GnssMeasurement measurement;
      measurement.stamp = stamp;
      measurement.raw_line = line;
      measurement.valid = false;
      measurement.reject_reason = "reject_invalid";
      measurements.push_back(measurement);
      break;
    }

    GnssMeasurement measurement;
    parseImuGnssJsonObject(json, line.substr(pos, next_pos - pos), stamp, measurement);
    measurements.push_back(measurement);
    pos = line.find(kPrefix, next_pos);
  }

  return measurements;
}

std::vector<GnssMeasurement> GnssManager::parseLine(const std::string &line, double receive_stamp) const
{
  const std::string trimmed = trimLine(line);
  std::vector<GnssMeasurement> out;
  if (trimmed.empty()) return out;

  if (!startsWith(trimmed, "$") &&
      !startsWith(trimmed, "#AGRICA,") &&
      !startsWith(trimmed, "@IMUGNSS:"))
  {
    size_t first = std::string::npos;
    const char *prefixes[] = {"$", "#AGRICA,", "@IMUGNSS:"};
    for (const char *prefix : prefixes)
    {
      const size_t pos = trimmed.find(prefix);
      if (pos != std::string::npos && (first == std::string::npos || pos < first)) first = pos;
    }
    if (first != std::string::npos) return parseLine(trimmed.substr(first), receive_stamp);
    return out;
  }

  if (startsWith(trimmed, "$"))
  {
    const size_t star = trimmed.find('*');
    if (star != std::string::npos && star + 3 < trimmed.size())
    {
      out = parseLine(trimmed.substr(0, star + 3), receive_stamp);
      const auto tail = parseLine(trimmed.substr(star + 3), receive_stamp);
      out.insert(out.end(), tail.begin(), tail.end());
      return out;
    }
  }

  if (startsWith(trimmed, "#AGRICA,"))
  {
    const size_t star = trimmed.find('*');
    if (star != std::string::npos && star + 9 < trimmed.size())
    {
      out = parseLine(trimmed.substr(0, star + 9), receive_stamp);
      const auto tail = parseLine(trimmed.substr(star + 9), receive_stamp);
      out.insert(out.end(), tail.begin(), tail.end());
      return out;
    }
  }

  auto modeAllows = [&](const std::string &message) {
    const std::string mode = upperString(parser_mode_);
    if (mode == "AUTO") return true;
    if (mode == "KSXT") return message == "KSXT";
    if (mode == "NMEA") return message == "GGA" || message == "RMC" ||
                               message == "GSA" || message == "GST" || message == "ZDA";
    if (mode == "AGRICA") return message == "AGRICA";
    if (mode == "LEGACY_IMUGNSS_JSON") return message == "LEGACY_IMUGNSS_JSON";
    return false;
  };

  auto baseMeasurement = [&](const std::string &source) {
    GnssMeasurement m;
    m.stamp = receive_stamp + time_offset_s_;
    m.receive_stamp = receive_stamp;
    m.source_message = source;
    m.raw_line = trimmed;
    m.reject_reason = "reject_parse";
    return m;
  };

  auto classifyKsxt = [&](int quality) {
    if (ksxt_fixed_quality_values_.count(quality)) return GnssSolutionType::RTK_FIXED;
    if (ksxt_float_quality_values_.count(quality)) return GnssSolutionType::RTK_FLOAT;
    if (ksxt_single_quality_values_.count(quality)) return GnssSolutionType::SINGLE;
    if (ksxt_invalid_quality_values_.count(quality)) return GnssSolutionType::INVALID;
    return GnssSolutionType::UNKNOWN;
  };

  auto classifyGga = [&](int quality) {
    if (gga_fixed_quality_values_.count(quality)) return GnssSolutionType::RTK_FIXED;
    if (gga_float_quality_values_.count(quality)) return GnssSolutionType::RTK_FLOAT;
    if (gga_differential_quality_values_.count(quality)) return GnssSolutionType::DIFFERENTIAL;
    if (gga_single_quality_values_.count(quality)) return GnssSolutionType::SINGLE;
    if (gga_invalid_quality_values_.count(quality)) return GnssSolutionType::INVALID;
    return GnssSolutionType::UNKNOWN;
  };

  auto classifyAgrica = [&](int position_type) {
    if (agrica_fixed_position_types_.count(position_type)) return GnssSolutionType::RTK_FIXED;
    if (agrica_float_position_types_.count(position_type)) return GnssSolutionType::RTK_FLOAT;
    if (agrica_manual_fixed_position_types_.count(position_type)) return GnssSolutionType::MANUAL_FIXED;
    if (agrica_differential_position_types_.count(position_type)) return GnssSolutionType::DIFFERENTIAL;
    if (agrica_single_position_types_.count(position_type)) return GnssSolutionType::SINGLE;
    if (agrica_invalid_position_types_.count(position_type)) return GnssSolutionType::INVALID;
    return GnssSolutionType::UNKNOWN;
  };

  auto finalizePosition = [&](GnssMeasurement &m) {
    m.state = m.raw_position_quality;
    if (m.solution_type == GnssSolutionType::INVALID)
    {
      m.valid = false;
      m.reject_reason = "reject_invalid_status";
      return;
    }
    if (!validLatLon(m.latitude_deg, m.longitude_deg))
    {
      m.valid = false;
      m.reject_reason = "reject_parse";
      return;
    }
    if (zeroLatLon(m.latitude_deg, m.longitude_deg))
    {
      m.valid = false;
      m.reject_reason = "reject_zero_position";
      return;
    }
    if (fixed_only_ && m.solution_type != GnssSolutionType::RTK_FIXED)
    {
      m.valid = true;
      m.reject_reason = "reject_not_fixed";
      return;
    }
    m.valid = true;
    m.reject_reason = m.source_message == "GGA" ? "accept_gga_fallback" : "accept_ksxt";
  };

  if (startsWith(trimmed, "@IMUGNSS:"))
  {
    if (!modeAllows("LEGACY_IMUGNSS_JSON")) return out;
    out = parseImuGnssJsonLine(trimmed, receive_stamp + time_offset_s_);
    return out;
  }

  if (startsWith(trimmed, "$"))
  {
    std::vector<std::string> fields;
    bool checksum_valid = false;
    const bool parsed_nmea = parseNmeaFields(trimmed, fields, checksum_valid);
    std::string source = "NMEA";
    if (!fields.empty())
    {
      const std::string talker = upperString(fields[0]);
      if (talker == "$KSXT") source = "KSXT";
      else if (talker == "$GNGGA" || talker == "$GPGGA") source = "GGA";
      else if (talker == "$GNRMC" || talker == "$GPRMC") source = "RMC";
      else if (talker == "$GPGSA" || talker == "$GNGSA") source = "GSA";
      else if (talker == "$GPGST" || talker == "$GNGST") source = "GST";
      else if (talker == "$GPZDA" || talker == "$GNZDA") source = "ZDA";
    }
    else if (startsWith(trimmed, "$KSXT,")) source = "KSXT";
    else if (startsWith(trimmed, "$GNGGA,") || startsWith(trimmed, "$GPGGA,")) source = "GGA";
    else if (startsWith(trimmed, "$GNRMC,") || startsWith(trimmed, "$GPRMC,")) source = "RMC";
    else if (startsWith(trimmed, "$GPGSA,") || startsWith(trimmed, "$GNGSA,")) source = "GSA";
    else if (startsWith(trimmed, "$GPGST,") || startsWith(trimmed, "$GNGST,")) source = "GST";
    else if (startsWith(trimmed, "$GPZDA,") || startsWith(trimmed, "$GNZDA,")) source = "ZDA";

    if (!modeAllows(source)) return out;

    GnssMeasurement m = baseMeasurement(source);
    m.checksum_valid = checksum_valid;
    if (!parsed_nmea || fields.empty())
    {
      m.reject_reason = "reject_parse";
      out.push_back(m);
      return out;
    }
    if (!checksum_valid)
    {
      m.reject_reason = "reject_checksum";
      out.push_back(m);
      return out;
    }

    if (source == "KSXT")
    {
      if (fields.size() < 22)
      {
        m.reject_reason = "reject_field_count";
        out.push_back(m);
        return out;
      }
      m.device_time_valid = parseKsxtUtc(fields[1], m.device_stamp);
      parseDoubleField(fields[2], m.longitude_deg);
      parseDoubleField(fields[3], m.latitude_deg);
      parseDoubleField(fields[4], m.altitude_m);
      parseDoubleField(fields[5], m.yaw_deg);
      parseDoubleField(fields[6], m.pitch_deg);
      double speed_kmh = 0.0;
      if (parseDoubleField(fields[8], speed_kmh))
      {
        // ponytail: scalar KSXT speed is logged only; velocity update waits for a real motion model.
      }
      parseDoubleField(fields[9], m.roll_deg);
      if (!parseIntField(fields[10], m.raw_position_quality))
      {
        m.reject_reason = "reject_parse";
        out.push_back(m);
        return out;
      }
      parseIntField(fields[11], m.raw_heading_quality);
      int slave_svs = 0, master_svs = 0;
      if (parseIntField(fields[12], slave_svs)) m.satellite_count += slave_svs;
      if (parseIntField(fields[13], master_svs)) m.satellite_count += master_svs;
      parseDoubleField(fields[17], m.velocity_east);
      parseDoubleField(fields[18], m.velocity_north);
      parseDoubleField(fields[19], m.velocity_up);
      m.solution_type = classifyKsxt(m.raw_position_quality);
      finalizePosition(m);
      out.push_back(m);
      return out;
    }

    if (source == "GGA")
    {
      if (fields.size() < 15)
      {
        m.reject_reason = "reject_field_count";
        out.push_back(m);
        return out;
      }
      parseNmeaLatLon(fields[2], fields[3], true, m.latitude_deg);
      parseNmeaLatLon(fields[4], fields[5], false, m.longitude_deg);
      if (!parseIntField(fields[6], m.raw_position_quality))
      {
        m.reject_reason = "reject_parse";
        out.push_back(m);
        return out;
      }
      parseIntField(fields[7], m.satellite_count);
      parseDoubleField(fields[8], m.hdop);
      parseDoubleField(fields[9], m.altitude_m);
      parseDoubleField(fields[13], m.differential_age_s);
      m.solution_type = classifyGga(m.raw_position_quality);
      finalizePosition(m);
      out.push_back(m);
      return out;
    }

    if (source == "RMC")
    {
      if (fields.size() < 3)
      {
        m.reject_reason = "reject_field_count";
      }
      else
      {
        const std::string status = upperString(trimField(fields[2]));
        m.raw_position_quality = status == "A" ? 1 : 0;
        m.state = m.raw_position_quality;
        m.solution_type = status == "A" ? GnssSolutionType::UNKNOWN : GnssSolutionType::INVALID;
        m.reject_reason = status == "A" ? "ok" : "reject_invalid_status";
      }
      out.push_back(m);
      return out;
    }

    if (source == "GSA")
    {
      if (fields.size() < 3 || !parseIntField(fields[2], m.raw_position_quality))
      {
        m.reject_reason = "reject_field_count";
      }
      else
      {
        m.state = m.raw_position_quality;
        m.solution_type = m.raw_position_quality >= 2 ? GnssSolutionType::UNKNOWN : GnssSolutionType::INVALID;
        m.reject_reason = m.raw_position_quality >= 2 ? "ok" : "reject_invalid_status";
      }
      out.push_back(m);
      return out;
    }

    if (source == "GST")
    {
      if (fields.size() >= 9)
      {
        parseDoubleField(fields[6], m.horizontal_std_m);
        double lon_std = std::numeric_limits<double>::quiet_NaN();
        if (parseDoubleField(fields[7], lon_std) && std::isfinite(m.horizontal_std_m))
        {
          m.horizontal_std_m = std::max(m.horizontal_std_m, lon_std);
        }
        parseDoubleField(fields[8], m.vertical_std_m);
        m.reject_reason = "ok";
      }
      else
      {
        m.reject_reason = "reject_field_count";
      }
      out.push_back(m);
      return out;
    }

    if (source == "ZDA")
    {
      m.reject_reason = fields.size() >= 5 && !trimField(fields[1]).empty() ? "ok" : "reject_parse";
      out.push_back(m);
      return out;
    }

    m.reject_reason = "reject_parse";
    out.push_back(m);
    return out;
  }

  if (startsWith(trimmed, "#AGRICA,"))
  {
    if (!modeAllows("AGRICA")) return out;
    GnssMeasurement m = baseMeasurement("AGRICA");
    const size_t semicolon = trimmed.find(';');
    const size_t star = trimmed.rfind('*');
    if (semicolon == std::string::npos || star == std::string::npos || star <= semicolon)
    {
      m.reject_reason = "reject_parse";
      out.push_back(m);
      return out;
    }
    m.checksum_valid = !agrica_crc_check_en_;
    if (agrica_crc_check_en_)
    {
      m.reject_reason = "reject_checksum";
      out.push_back(m);
      return out;
    }
    const std::string data_part = trimmed.substr(semicolon + 1, star - semicolon - 1);
    const std::vector<std::string> data = splitPreserveEmpty(data_part, ',');
    if (data.size() < 55 || data[0] != "GNSS")
    {
      m.reject_reason = "reject_field_count";
      out.push_back(m);
      return out;
    }
    if (!parseIntField(data[8], m.raw_position_quality))
    {
      m.reject_reason = "reject_parse";
      out.push_back(m);
      return out;
    }
    m.state = m.raw_position_quality;
    parseIntField(data[10], m.satellite_count);
    int bds_svs = 0, glo_svs = 0, gal_svs = 0;
    if (parseIntField(data[11], bds_svs)) m.satellite_count += bds_svs;
    if (parseIntField(data[12], glo_svs)) m.satellite_count += glo_svs;
    if (parseIntField(data[53], gal_svs)) m.satellite_count += gal_svs;
    parseDoubleField(data[19], m.yaw_deg);
    parseDoubleField(data[20], m.pitch_deg);
    parseDoubleField(data[21], m.roll_deg);
    parseDoubleField(data[24], m.velocity_east);
    parseDoubleField(data[23], m.velocity_north);
    parseDoubleField(data[25], m.velocity_up);
    parseDoubleField(data[29], m.latitude_deg);
    parseDoubleField(data[30], m.longitude_deg);
    parseDoubleField(data[31], m.altitude_m);
    parseDoubleField(data[35], m.horizontal_std_m);
    double lon_std = std::numeric_limits<double>::quiet_NaN();
    if (parseDoubleField(data[36], lon_std) && std::isfinite(m.horizontal_std_m))
    {
      m.horizontal_std_m = std::max(m.horizontal_std_m, lon_std);
    }
    parseDoubleField(data[37], m.vertical_std_m);
    parseDoubleField(data[48], m.differential_age_s);
    m.solution_type = classifyAgrica(m.raw_position_quality);
    finalizePosition(m);
    if (m.valid) m.reject_reason = "ok";
    out.push_back(m);
    return out;
  }

  return out;
}

GnssManager::GnssManager() = default;

GnssManager::~GnssManager()
{
  shutdown();
}

bool GnssManager::initialize(ros::NodeHandle &nh, const std::string &save_path)
{
  if (!loadParameters(nh) || !en_)
  {
    convergence_state_ = ConvergenceState::DISABLED;
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    raw_log_file_.open(save_path + raw_log_filename_, std::ios::out | std::ios::app);
    parsed_log_file_.open(save_path + parsed_log_filename_, std::ios::out | std::ios::app);
    update_log_file_.open(save_path + update_log_filename_, std::ios::out | std::ios::app);

    if (raw_log_file_.is_open())
    {
      raw_log_file_ << "# stamp raw_line\n";
      raw_log_file_.flush();
    }
    else
    {
      ROS_WARN("[GNSS] Failed to open raw log: %s", (save_path + raw_log_filename_).c_str());
    }

    if (parsed_log_file_.is_open())
    {
      parsed_log_file_ << "# receive_stamp message_type checksum_valid device_time_valid latitude longitude altitude "
                       << "raw_quality solution_type satellite_count hdop horizontal_std vertical_std "
                       << "differential_age valid reject_reason raw_line\n";
      parsed_log_file_.flush();
    }
    else
    {
      ROS_WARN("[GNSS] Failed to open parsed log: %s", (save_path + parsed_log_filename_).c_str());
    }

    if (update_log_file_.is_open())
    {
      update_log_file_ << "# stamp seq device_state convergence_state lat lon alt "
                       << "source enu_x enu_y enu_z world_x world_y world_z pred_x pred_y "
                       << "residual_x residual_y residual_norm sigma_xy mahalanobis time_diff "
                       << "correction_raw correction_applied action\n";
      update_log_file_.flush();
    }
    else
    {
      ROS_WARN("[GNSS] Failed to open update log: %s", (save_path + update_log_filename_).c_str());
    }
  }

  const std::string source = toLower(input_source_);
  if (source != "serial")
  {
    ROS_WARN("[GNSS] gps/source='%s' is not implemented in this build. Falling back to serial.", input_source_.c_str());
  }

  if (!openSerial())
  {
    ROS_WARN("[GNSS] Serial reader will keep retrying because %s could not be opened/configured.", serial_port_.c_str());
    transitionTo(ConvergenceState::DEGRADED, ros::Time::now().toSec(), "GNSS_SERIAL_OPEN_FAILED");
  }
  else
  {
    transitionTo(ConvergenceState::SERIAL_OPENED, ros::Time::now().toSec(), "GNSS_SERIAL_OPENED");
    transitionTo(ConvergenceState::WAIT_VALID_DATA, ros::Time::now().toSec(), "WAIT_GNSS_VALID_DATA");
  }

  running_.store(true);
  read_thread_ = std::thread(&GnssManager::readLoop, this);
  ROS_INFO("[GNSS] Serial reader thread started: port=%s baud=%d DTR=%d RTS=%d update_en=%d",
           serial_port_.c_str(), baudrate_, static_cast<int>(dtr_high_),
           static_cast<int>(rts_high_), static_cast<int>(update_en_));
  return true;
}

void GnssManager::shutdown()
{
  running_.store(false);
  if (read_thread_.joinable()) read_thread_.join();
  closeSerial();

  std::lock_guard<std::mutex> lock(log_mutex_);
  if (raw_log_file_.is_open())
  {
    raw_log_file_.flush();
    raw_log_file_.close();
  }
  if (parsed_log_file_.is_open())
  {
    parsed_log_file_.flush();
    parsed_log_file_.close();
  }
  if (update_log_file_.is_open())
  {
    update_log_file_.flush();
    update_log_file_.close();
  }
}

bool GnssManager::loadParameters(ros::NodeHandle &nh)
{
  nh.param<bool>("gps/en", en_, false);
  nh.param<bool>("gps/enable", en_, en_);
  bool update_en_primary = update_en_;
  bool update_en_alias = update_en_;
  bool use_gps_position_legacy = update_en_;
  const bool has_update_en = nh.getParam("gps/update_en", update_en_primary);
  const bool has_update_enable = nh.getParam("gps/update_enable", update_en_alias);
  const bool has_use_gps_position = nh.getParam("gps/use_gps_position", use_gps_position_legacy);
  if (has_update_en)
  {
    update_en_ = update_en_primary;
  }
  else if (has_update_enable)
  {
    update_en_ = update_en_alias;
  }
  else if (has_use_gps_position)
  {
    update_en_ = use_gps_position_legacy;
  }
  if (has_update_en && has_use_gps_position && update_en_primary != use_gps_position_legacy)
  {
    ROS_WARN("[GNSS] gps/update_en=%d overrides legacy gps/use_gps_position=%d.",
             static_cast<int>(update_en_primary), static_cast<int>(use_gps_position_legacy));
  }

  nh.param<std::string>("gps/source", input_source_, "serial");
  nh.param<std::string>("gps/serial_port", serial_port_, "/dev/ttyUSB0");
  nh.param<int>("gps/baudrate", baudrate_, 921600);
  nh.param<bool>("gps/dtr", dtr_high_, true);
  nh.param<bool>("gps/rts", rts_high_, false);
  nh.param<std::string>("gps/parser_mode", parser_mode_, "auto");
  nh.param<std::string>("gps/primary_position_message", primary_position_message_, "KSXT");
  nh.param<std::string>("gps/fallback_position_message", fallback_position_message_, "GGA");
  nh.param<double>("gps/time_offset_s", time_offset_s_, 0.0);
  nh.param<double>("gps/match_threshold_s", match_threshold_s_, 0.10);
  nh.param<double>("gps/stale_timeout_s", stale_timeout_s_, 0.50);
  nh.param<int>("gps/max_queue_size", max_queue_size_, 512);

  nh.param<double>("gps/startup_convergence_s", startup_convergence_s_, 30.0);
  nh.param<int>("gps/fixed_confirm_count", fixed_confirm_count_, 10);
  nh.param<int>("gps/reacquire_confirm_count", reacquire_confirm_count_, 5);
  nh.param<bool>("gps/reset_convergence_on_long_stale", reset_convergence_on_long_stale_, false);
  nh.param<double>("gps/reset_convergence_stale_s", reset_convergence_stale_s_, 5.0);
  loadIntSetParam(nh, "gps/fixed_state_values", fixed_state_values_);
  loadIntSetParam(nh, "gps/float_state_values", float_state_values_);
  loadIntSetParam(nh, "gps/invalid_state_values", invalid_state_values_);
  loadIntSetParam(nh, "gps/ksxt/invalid_quality_values", ksxt_invalid_quality_values_);
  loadIntSetParam(nh, "gps/ksxt/single_quality_values", ksxt_single_quality_values_);
  loadIntSetParam(nh, "gps/ksxt/float_quality_values", ksxt_float_quality_values_);
  loadIntSetParam(nh, "gps/ksxt/fixed_quality_values", ksxt_fixed_quality_values_);
  loadIntSetParam(nh, "gps/gga/invalid_quality_values", gga_invalid_quality_values_);
  loadIntSetParam(nh, "gps/gga/single_quality_values", gga_single_quality_values_);
  loadIntSetParam(nh, "gps/gga/differential_quality_values", gga_differential_quality_values_);
  loadIntSetParam(nh, "gps/gga/fixed_quality_values", gga_fixed_quality_values_);
  loadIntSetParam(nh, "gps/gga/float_quality_values", gga_float_quality_values_);
  loadIntSetParam(nh, "gps/agrica/invalid_position_types", agrica_invalid_position_types_);
  loadIntSetParam(nh, "gps/agrica/single_position_types", agrica_single_position_types_);
  loadIntSetParam(nh, "gps/agrica/differential_position_types", agrica_differential_position_types_);
  loadIntSetParam(nh, "gps/agrica/fixed_position_types", agrica_fixed_position_types_);
  loadIntSetParam(nh, "gps/agrica/float_position_types", agrica_float_position_types_);
  loadIntSetParam(nh, "gps/agrica/manual_fixed_position_types", agrica_manual_fixed_position_types_);
  nh.param<bool>("gps/agrica/crc_check_en", agrica_crc_check_en_, false);
  nh.param<bool>("gps/fixed_only", fixed_only_, true);

  nh.param<std::string>("gps/origin_mode", origin_mode_, "first_fixed");
  nh.param<double>("gps/origin_latitude_deg", origin_latitude_deg_, 0.0);
  nh.param<double>("gps/origin_longitude_deg", origin_longitude_deg_, 0.0);
  nh.param<double>("gps/origin_altitude_m", origin_altitude_m_, 0.0);
  nh.param<std::string>("gps/altitude_type", altitude_type_, "ellipsoid");
  nh.param<double>("gps/geoid_separation_m", geoid_separation_m_, 0.0);

  nh.param<bool>("gps/frame_align_en", frame_align_en_, true);
  nh.param<std::string>("gps/frame_align_mode", frame_align_mode_, "trajectory_2d");
  nh.param<int>("gps/frame_align_min_samples", frame_align_min_samples_, 20);
  nh.param<double>("gps/frame_align_min_motion_m", frame_align_min_motion_m_, 10.0);
  nh.param<double>("gps/frame_align_max_rms_m", frame_align_max_rms_m_, 0.50);
  nh.param<double>("gps/frame_align_max_error_m", frame_align_max_error_m_, 1.50);
  nh.param<bool>("gps/frame_align_freeze_after_success", frame_align_freeze_after_success_, true);
  nh.param<double>("gps/frame_align_yaw_deg", frame_align_yaw_deg_, 0.0);
  if (!loadVec3Param(nh, "gps/frame_align_translation", frame_align_translation_))
  {
    frame_align_translation_.setZero();
  }

  nh.param<bool>("gps/update_xy_only", update_xy_only_, true);
  nh.param<bool>("gps/update_z", update_z_, false);
  nh.param<bool>("gps/update_orientation", update_orientation_, false);
  nh.param<double>("gps/sigma_xy_fixed_m", sigma_xy_fixed_m_, 0.10);
  nh.param<double>("gps/sigma_z_fixed_m", sigma_z_fixed_m_, 0.30);
  nh.param<double>("gps/position_cov_floor_m", position_cov_floor_m_, 0.20);
  nh.param<double>("gps/chi2_gate_2d", chi2_gate_2d_, 9.21);
  nh.param<double>("gps/max_residual_m", max_residual_m_, 3.0);
  nh.param<double>("gps/max_update_step_m", max_update_step_m_, 0.20);
  if (!loadVec3Param(nh, "gps/lever_arm_body_to_gnss", lever_arm_body_to_gnss_))
  {
    lever_arm_body_to_gnss_.setZero();
  }

  nh.param<int>("gps/pause_map_update_frames", pause_map_update_frames_, 3);
  nh.param<double>("gps/pause_map_update_min_correction_m", pause_map_update_min_correction_m_, 0.05);

  nh.param<std::string>("gps/raw_log_filename", raw_log_filename_, "gnss_raw.txt");
  nh.param<std::string>("gps/parsed_log_filename", parsed_log_filename_, "gnss_parsed.txt");
  nh.param<std::string>("gps/update_log_filename", update_log_filename_, "gnss_updates.txt");
  nh.param<int>("gps/log_flush_stride", log_flush_stride_, 1);

  parser_mode_ = toLower(parser_mode_);
  primary_position_message_ = upperString(primary_position_message_);
  fallback_position_message_ = upperString(fallback_position_message_);
  input_source_ = toLower(input_source_);
  origin_mode_ = toLower(origin_mode_);
  altitude_type_ = toLower(altitude_type_);
  frame_align_mode_ = toLower(frame_align_mode_);
  std::replace(origin_mode_.begin(), origin_mode_.end(), '-', '_');
  std::replace(altitude_type_.begin(), altitude_type_.end(), '-', '_');
  std::replace(frame_align_mode_.begin(), frame_align_mode_.end(), '-', '_');

  if (parser_mode_ != "auto" &&
      parser_mode_ != "ksxt" &&
      parser_mode_ != "nmea" &&
      parser_mode_ != "agrica" &&
      parser_mode_ != "legacy_imugnss_json")
  {
    ROS_WARN("[GNSS] Unknown gps/parser_mode='%s'. Use auto.", parser_mode_.c_str());
    parser_mode_ = "auto";
  }
  if (primary_position_message_ != "KSXT" && primary_position_message_ != "GGA")
  {
    ROS_WARN("[GNSS] Unsupported gps/primary_position_message='%s'. Use KSXT.", primary_position_message_.c_str());
    primary_position_message_ = "KSXT";
  }
  if (fallback_position_message_ != "KSXT" && fallback_position_message_ != "GGA")
  {
    ROS_WARN("[GNSS] Unsupported gps/fallback_position_message='%s'. Use GGA.", fallback_position_message_.c_str());
    fallback_position_message_ = "GGA";
  }
  if (origin_mode_ != "first_fixed" && origin_mode_ != "manual")
  {
    ROS_WARN("[GNSS] Unknown gps/origin_mode='%s'. Use first_fixed.", origin_mode_.c_str());
    origin_mode_ = "first_fixed";
  }
  if (altitude_type_ != "ellipsoid" && altitude_type_ != "orthometric")
  {
    ROS_WARN("[GNSS] Unknown gps/altitude_type='%s'. Use ellipsoid.", altitude_type_.c_str());
    altitude_type_ = "ellipsoid";
  }
  if (frame_align_mode_ != "trajectory_2d" && frame_align_mode_ != "manual")
  {
    ROS_WARN("[GNSS] Unknown gps/frame_align_mode='%s'. Use trajectory_2d.", frame_align_mode_.c_str());
    frame_align_mode_ = "trajectory_2d";
  }

  baudrate_ = std::max(1, baudrate_);
  match_threshold_s_ = std::max(0.0, match_threshold_s_);
  stale_timeout_s_ = std::max(0.0, stale_timeout_s_);
  max_queue_size_ = std::max(8, max_queue_size_);
  startup_convergence_s_ = std::max(0.0, startup_convergence_s_);
  fixed_confirm_count_ = std::max(1, fixed_confirm_count_);
  reacquire_confirm_count_ = std::max(1, reacquire_confirm_count_);
  reset_convergence_stale_s_ = std::max(stale_timeout_s_, reset_convergence_stale_s_);
  frame_align_min_samples_ = std::max(2, frame_align_min_samples_);
  frame_align_min_motion_m_ = std::max(0.0, frame_align_min_motion_m_);
  frame_align_max_rms_m_ = std::max(0.0, frame_align_max_rms_m_);
  frame_align_max_error_m_ = std::max(0.0, frame_align_max_error_m_);
  sigma_xy_fixed_m_ = std::max(1e-3, sigma_xy_fixed_m_);
  sigma_z_fixed_m_ = std::max(1e-3, sigma_z_fixed_m_);
  position_cov_floor_m_ = std::max(0.0, position_cov_floor_m_);
  chi2_gate_2d_ = std::max(0.0, chi2_gate_2d_);
  max_residual_m_ = std::max(0.0, max_residual_m_);
  max_update_step_m_ = std::max(0.0, max_update_step_m_);
  pause_map_update_frames_ = std::max(0, pause_map_update_frames_);
  pause_map_update_min_correction_m_ = std::max(0.0, pause_map_update_min_correction_m_);
  log_flush_stride_ = std::max(1, log_flush_stride_);

  frame_align_yaw_rad_ = frame_align_yaw_deg_ * kDegToRad;
  frame_align_t_ = frame_align_translation_;
  if (!frame_align_en_)
  {
    frame_aligned_ = true;
    frame_align_yaw_rad_ = 0.0;
    frame_align_t_.setZero();
  }
  else if (frame_align_mode_ == "manual")
  {
    frame_aligned_ = true;
    frame_align_t_ = frame_align_translation_;
  }

  if (origin_mode_ == "manual")
  {
    if (origin_latitude_deg_ < -90.0 || origin_latitude_deg_ > 90.0 ||
        origin_longitude_deg_ < -180.0 || origin_longitude_deg_ > 180.0)
    {
      ROS_WARN("[GNSS] Manual origin lat/lon are invalid. GNSS updates will wait for a valid origin config.");
      origin_ready_ = false;
    }
    else
    {
      const double origin_h =
          altitude_type_ == "orthometric" ? origin_altitude_m_ + geoid_separation_m_ : origin_altitude_m_;
      origin_ecef_ = geodeticToEcef(origin_latitude_deg_, origin_longitude_deg_, origin_h);
      origin_ready_ = true;
    }
  }

  ROS_INFO("[GNSS] enable=%d update_en=%d serial=%s baud=%d parser=%s primary=%s fallback=%s fixed_only=%d origin_mode=%s frame_align_mode=%s",
           static_cast<int>(en_), static_cast<int>(update_en_), serial_port_.c_str(), baudrate_,
           parser_mode_.c_str(), primary_position_message_.c_str(), fallback_position_message_.c_str(),
           static_cast<int>(fixed_only_), origin_mode_.c_str(), frame_align_mode_.c_str());
  return true;
}

bool GnssManager::openSerial()
{
  closeSerial();
  serial_fd_ = ::open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (serial_fd_ < 0)
  {
    serial_opened_.store(false);
    ROS_WARN_THROTTLE(5.0, "[GNSS] Failed to open %s: %s", serial_port_.c_str(), std::strerror(errno));
    return false;
  }
  if (!configureSerial())
  {
    closeSerial();
    serial_opened_.store(false);
    return false;
  }
  serial_opened_.store(true);
  return true;
}

bool GnssManager::configureSerial()
{
  if (serial_fd_ < 0) return false;

  termios tty;
  if (::tcgetattr(serial_fd_, &tty) != 0)
  {
    ROS_WARN("[GNSS] tcgetattr failed on %s: %s", serial_port_.c_str(), std::strerror(errno));
    return false;
  }

  speed_t speed = B115200;
  if (!baudrateToTermios(baudrate_, speed))
  {
    ROS_ERROR("[GNSS] Unsupported baudrate=%d with this termios build. Set gps/baudrate to a supported value such as 115200 or 921600, or add termios2/BOTHER support for this platform.",
              baudrate_);
    return false;
  }

  cfmakeraw(&tty);
  ::cfsetispeed(&tty, speed);
  ::cfsetospeed(&tty, speed);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;
  tty.c_cflag &= ~PARENB;
  tty.c_cflag &= ~CSTOPB;
#ifdef CRTSCTS
  tty.c_cflag &= ~CRTSCTS;
#endif
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 1;

  if (::tcsetattr(serial_fd_, TCSANOW, &tty) != 0)
  {
    ROS_WARN("[GNSS] tcsetattr failed on %s: %s", serial_port_.c_str(), std::strerror(errno));
    return false;
  }
  ::tcflush(serial_fd_, TCIOFLUSH);

  int modem_bits = 0;
  if (::ioctl(serial_fd_, TIOCMGET, &modem_bits) == 0)
  {
    if (dtr_high_) modem_bits |= TIOCM_DTR;
    else modem_bits &= ~TIOCM_DTR;
    if (rts_high_) modem_bits |= TIOCM_RTS;
    else modem_bits &= ~TIOCM_RTS;
    if (::ioctl(serial_fd_, TIOCMSET, &modem_bits) != 0)
    {
      ROS_WARN("[GNSS] Failed to set DTR/RTS on %s: %s", serial_port_.c_str(), std::strerror(errno));
    }
  }
  else
  {
    ROS_WARN("[GNSS] Failed to read modem bits on %s: %s", serial_port_.c_str(), std::strerror(errno));
  }

  return true;
}

void GnssManager::closeSerial()
{
  if (serial_fd_ >= 0)
  {
    ::close(serial_fd_);
    serial_fd_ = -1;
  }
  serial_opened_.store(false);
}

void GnssManager::readLoop()
{
  std::string line_buffer;
  line_buffer.reserve(1024);
  char buffer[512];
  double last_reopen_attempt = 0.0;

  while (running_.load())
  {
    if (serial_fd_ < 0)
    {
      const double now = ros::Time::now().toSec();
      if (now - last_reopen_attempt > 1.0)
      {
        last_reopen_attempt = now;
        if (openSerial())
        {
          transitionTo(ConvergenceState::SERIAL_OPENED, now, "GNSS_SERIAL_REOPENED");
          transitionTo(ConvergenceState::WAIT_VALID_DATA, now, "WAIT_GNSS_VALID_DATA");
        }
      }
      ros::Duration(0.05).sleep();
      continue;
    }

    const ssize_t n = ::read(serial_fd_, buffer, sizeof(buffer));
    if (n > 0)
    {
      for (ssize_t i = 0; i < n; ++i)
      {
        const char c = buffer[i];
        if (c == '\n' || c == '\r')
        {
          const std::string line = trimLine(line_buffer);
          line_buffer.clear();
          if (!line.empty()) handleLine(line, ros::Time::now().toSec());
        }
        else
        {
          line_buffer.push_back(c);
          if (line_buffer.size() > 8192)
          {
            const std::string line = trimLine(line_buffer);
            line_buffer.clear();
            if (!line.empty()) handleLine(line, ros::Time::now().toSec());
          }
        }
      }
    }
    else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
    {
      ROS_WARN_THROTTLE(5.0, "[GNSS] Serial read error on %s: %s", serial_port_.c_str(), std::strerror(errno));
      closeSerial();
      transitionTo(ConvergenceState::DEGRADED, ros::Time::now().toSec(), "GNSS_SERIAL_READ_ERROR");
      ros::Duration(0.05).sleep();
    }
    else
    {
      ros::Duration(0.005).sleep();
    }
  }
}

void GnssManager::handleLine(const std::string &line, double stamp)
{
  logRawLine(stamp, line);

  auto measurements = parseLine(line, stamp);
  if (measurements.empty()) return;

  const std::string primary = upperString(primary_position_message_);
  const std::string fallback = upperString(fallback_position_message_);
  std::vector<GnssMeasurement> queued;
  {
    std::lock_guard<std::mutex> lock(measurement_mutex_);
    for (auto &measurement : measurements)
    {
      if (measurement.source_message == "LEGACY_IMUGNSS_JSON" && measurement.seq >= 0)
      {
        if (have_last_seq_ && measurement.seq <= last_seq_)
        {
          measurement.valid = false;
          measurement.reject_reason = "reject_duplicate";
        }
        else
        {
          have_last_seq_ = true;
          last_seq_ = measurement.seq;
        }
      }

      const std::string source = upperString(measurement.source_message);
      const bool position_candidate =
          source == primary || source == fallback || source == "LEGACY_IMUGNSS_JSON";
      if (position_candidate && measurement.checksum_valid &&
          (measurement.valid || measurement.reject_reason == "reject_duplicate"))
      {
        measurement_queue_.push_back(measurement);
        queued.push_back(measurement);
        last_measurement_stamp_ = measurement.stamp;
        have_last_measurement_stamp_ = true;
      }
    }
    while (static_cast<int>(measurement_queue_.size()) > max_queue_size_)
    {
      measurement_queue_.pop_front();
    }
  }

  (void)queued;
  for (const auto &measurement : measurements) logParsedMeasurement(measurement);
}

bool GnssManager::takeLatestMeasurement(double current_lidar_stamp, GnssMeasurement &measurement, double &time_diff_s)
{
  std::lock_guard<std::mutex> lock(measurement_mutex_);
  if (measurement_queue_.empty()) return false;

  const std::string primary = upperString(primary_position_message_);
  const std::string fallback = upperString(fallback_position_message_);
  bool have_primary = false;
  bool have_primary_usable = false;
  bool have_fallback = false;
  bool have_fallback_usable = false;
  bool have_other = false;
  GnssMeasurement latest_primary;
  GnssMeasurement latest_primary_usable;
  GnssMeasurement latest_fallback;
  GnssMeasurement latest_fallback_usable;
  GnssMeasurement latest_other;
  while (!measurement_queue_.empty())
  {
    const GnssMeasurement candidate = measurement_queue_.front();
    measurement_queue_.pop_front();
    const std::string source = upperString(candidate.source_message);
    const bool usable_for_update = candidate.valid && isFixedSolution(candidate);
    if (source == primary)
    {
      if (!have_primary || candidate.stamp >= latest_primary.stamp)
      {
        latest_primary = candidate;
        have_primary = true;
      }
      if (usable_for_update && (!have_primary_usable || candidate.stamp >= latest_primary_usable.stamp))
      {
        latest_primary_usable = candidate;
        have_primary_usable = true;
      }
    }
    else if (source == fallback)
    {
      if (!have_fallback || candidate.stamp >= latest_fallback.stamp)
      {
        latest_fallback = candidate;
        have_fallback = true;
      }
      if (usable_for_update && (!have_fallback_usable || candidate.stamp >= latest_fallback_usable.stamp))
      {
        latest_fallback_usable = candidate;
        have_fallback_usable = true;
      }
    }
    else if (!have_other || candidate.stamp >= latest_other.stamp)
    {
      latest_other = candidate;
      have_other = true;
    }
  }

  if (have_primary_usable)
  {
    measurement = latest_primary_usable;
    measurement.reject_reason = "accept_ksxt";
  }
  else if (have_fallback_usable)
  {
    measurement = latest_fallback_usable;
    measurement.reject_reason = "accept_gga_fallback";
  }
  else if (have_primary)
  {
    measurement = latest_primary;
  }
  else if (have_fallback)
  {
    measurement = latest_fallback;
  }
  else if (have_other)
  {
    measurement = latest_other;
  }
  else
  {
    return false;
  }

  const double epoch_stamp = measurement.device_time_valid ? measurement.device_stamp : measurement.stamp;
  if (have_last_update_epoch_ &&
      std::fabs(epoch_stamp - last_update_epoch_stamp_) < 1e-3 &&
      upperString(measurement.source_message) == upperString(last_update_epoch_source_))
  {
    measurement.valid = false;
    measurement.reject_reason = "reject_duplicate_epoch";
  }

  const double reference_stamp = current_lidar_stamp > 0.0 ? current_lidar_stamp : ros::Time::now().toSec();
  time_diff_s = measurement.stamp - reference_stamp;
  return true;
}

bool GnssManager::isFixedState(int state) const
{
  return fixed_state_values_.find(state) != fixed_state_values_.end();
}

bool GnssManager::isFixedSolution(const GnssMeasurement &measurement) const
{
  if (measurement.source_message == "LEGACY_IMUGNSS_JSON") return isFixedState(measurement.state);
  return measurement.solution_type == GnssSolutionType::RTK_FIXED ||
         measurement.solution_type == GnssSolutionType::MANUAL_FIXED;
}

bool GnssManager::isInvalidState(int state) const
{
  return invalid_state_values_.find(state) != invalid_state_values_.end();
}

double GnssManager::ellipsoidHeight(const GnssMeasurement &measurement) const
{
  if (altitude_type_ == "orthometric") return measurement.altitude_m + geoid_separation_m_;
  return measurement.altitude_m;
}

bool GnssManager::ensureOrigin(const GnssMeasurement &measurement)
{
  if (origin_ready_) return true;
  if (origin_mode_ != "first_fixed") return false;

  origin_latitude_deg_ = measurement.latitude_deg;
  origin_longitude_deg_ = measurement.longitude_deg;
  origin_altitude_m_ = ellipsoidHeight(measurement);
  origin_ecef_ = geodeticToEcef(origin_latitude_deg_, origin_longitude_deg_, origin_altitude_m_);
  origin_ready_ = true;
  std::ostringstream oss;
  oss << "GNSS_ORIGIN_SET mode=first_fixed lat=" << origin_latitude_deg_
      << " lon=" << origin_longitude_deg_
      << " ellipsoid_height=" << origin_altitude_m_;
  logEventThrottled(measurement.stamp, "origin_set", 0.0, "INFO", oss.str());
  return true;
}

bool GnssManager::convertMeasurement(const GnssMeasurement &measurement, V3D &enu, V3D &world)
{
  if (!origin_ready_) return false;
  const V3D ecef = geodeticToEcef(measurement.latitude_deg,
                                  measurement.longitude_deg,
                                  ellipsoidHeight(measurement));
  enu = ecefToEnu(ecef, origin_ecef_, origin_latitude_deg_, origin_longitude_deg_);
  if (!finiteVec3(enu)) return false;
  world = enuToWorld(enu);
  return finiteVec3(world);
}

bool GnssManager::updateConvergenceAndAlignment(const GnssMeasurement &measurement,
                                                double time_diff_s,
                                                const StatesGroup &state,
                                                V3D &enu,
                                                V3D &world,
                                                std::string &reject_action)
{
  if (!measurement.valid)
  {
    reject_action = measurement.reject_reason.empty() ? "reject_invalid" : measurement.reject_reason;
    if (convergence_state_ == ConvergenceState::READY)
    {
      consecutive_fixed_count_ = 0;
      transitionTo(ConvergenceState::DEGRADED, measurement.stamp, "GNSS_FIXED_LOST");
    }
    return false;
  }

  if (measurement.solution_type == GnssSolutionType::INVALID ||
      (measurement.source_message == "LEGACY_IMUGNSS_JSON" && isInvalidState(measurement.state)))
  {
    reject_action = "reject_invalid";
    consecutive_fixed_count_ = 0;
    if (convergence_state_ == ConvergenceState::READY)
    {
      transitionTo(ConvergenceState::DEGRADED, measurement.stamp, "GNSS_FIXED_LOST");
    }
    return false;
  }

  if (!have_first_valid_stamp_)
  {
    have_first_valid_stamp_ = true;
    first_valid_stamp_ = measurement.stamp;
    transitionTo(ConvergenceState::WARMING_UP, measurement.stamp, "GNSS_WARMING_UP");
  }

  const double convergence_elapsed = measurement.stamp - first_valid_stamp_;
  if (convergence_elapsed < startup_convergence_s_)
  {
    reject_action = "reject_not_converged";
    transitionTo(ConvergenceState::WARMING_UP, measurement.stamp, "GNSS_WARMING_UP");
    return false;
  }

  const bool fixed = isFixedSolution(measurement);
  if (!fixed)
  {
    consecutive_fixed_count_ = 0;
    reject_action = "reject_not_fixed";
    if (convergence_state_ == ConvergenceState::READY)
    {
      transitionTo(ConvergenceState::DEGRADED, measurement.stamp, "GNSS_FIXED_LOST");
    }
    else
    {
      transitionTo(ConvergenceState::WAIT_FIXED, measurement.stamp, "WAIT_GNSS_FIXED");
    }
    return false;
  }

  consecutive_fixed_count_++;
  const int required_confirm_count = was_ready_once_ ? reacquire_confirm_count_ : fixed_confirm_count_;
  if (consecutive_fixed_count_ < required_confirm_count)
  {
    reject_action = "reject_not_fixed";
    transitionTo(ConvergenceState::WAIT_FIXED, measurement.stamp, "WAIT_GNSS_FIXED");
    return false;
  }

  logEventThrottled(measurement.stamp, "fixed_confirmed", 2.0, "INFO",
                    "GNSS_FIXED_CONFIRMED count=" + std::to_string(consecutive_fixed_count_));

  if (!ensureOrigin(measurement) || !convertMeasurement(measurement, enu, world))
  {
    reject_action = "reject_invalid";
    return false;
  }

  if (!frame_aligned_)
  {
    collectAlignSample(measurement, state, enu);
    if (!trySolveFrameAlignment())
    {
      reject_action = "reject_not_aligned";
      transitionTo(ConvergenceState::ALIGNING, measurement.stamp, "WAIT_GNSS_FRAME_ALIGN");
      return false;
    }
    world = enuToWorld(enu);
  }

  was_ready_once_ = true;
  transitionTo(ConvergenceState::READY, measurement.stamp, "GNSS_READY");
  (void)time_diff_s;
  return true;
}

void GnssManager::collectAlignSample(const GnssMeasurement &measurement,
                                     const StatesGroup &state,
                                     const V3D &enu)
{
  if (!frame_align_en_ || frame_align_mode_ != "trajectory_2d" || frame_aligned_) return;

  GnssFrameAlignSample sample;
  sample.enu_position = enu;
  sample.world_position = state.pos_end + state.rot_end * lever_arm_body_to_gnss_;
  sample.stamp = measurement.stamp;
  frame_align_samples_.push_back(sample);
  if (frame_align_samples_.size() > 400)
  {
    frame_align_samples_.erase(frame_align_samples_.begin(),
                               frame_align_samples_.begin() + static_cast<long>(frame_align_samples_.size() - 400));
  }
}

bool GnssManager::trySolveFrameAlignment()
{
  if (!frame_align_en_)
  {
    frame_aligned_ = true;
    return true;
  }
  if (frame_align_mode_ == "manual")
  {
    frame_aligned_ = true;
    frame_align_yaw_rad_ = frame_align_yaw_deg_ * kDegToRad;
    frame_align_t_ = frame_align_translation_;
    return true;
  }
  if (frame_aligned_ && frame_align_freeze_after_success_) return true;
  if (static_cast<int>(frame_align_samples_.size()) < frame_align_min_samples_) return false;
  if (sampleMotionExtent(frame_align_samples_) < frame_align_min_motion_m_)
  {
    logEventThrottled(ros::Time::now().toSec(), "frame_align_wait_motion", 3.0, "INFO",
                      "WAIT_GNSS_FRAME_ALIGN reason=motion_too_small samples=" +
                          std::to_string(frame_align_samples_.size()) +
                          " min_motion=" + std::to_string(frame_align_min_motion_m_));
    return false;
  }

  logEventThrottled(ros::Time::now().toSec(), "frame_align_solving", 2.0, "INFO",
                    "GNSS_FRAME_ALIGN_SOLVING samples=" + std::to_string(frame_align_samples_.size()));

  const int n = static_cast<int>(frame_align_samples_.size());
  int solve_count = n;
  if (n >= frame_align_min_samples_ + 5)
  {
    solve_count = std::max(frame_align_min_samples_, static_cast<int>(std::floor(n * 0.7)));
  }
  std::vector<GnssFrameAlignSample> solve_samples(frame_align_samples_.begin(),
                                                  frame_align_samples_.begin() + solve_count);
  std::vector<GnssFrameAlignSample> validation_samples;
  if (solve_count < n)
  {
    validation_samples.assign(frame_align_samples_.begin() + solve_count, frame_align_samples_.end());
  }

  double yaw_rad = 0.0;
  V3D translation = V3D::Zero();
  double rms = 0.0;
  double max_error = 0.0;
  if (!solveFrameAlignment(solve_samples, validation_samples, yaw_rad, translation, rms, max_error))
  {
    logEventThrottled(ros::Time::now().toSec(), "frame_align_reject_solve", 2.0, "WARN",
                      "GNSS_FRAME_ALIGN_REJECT reason=solve_failed samples=" + std::to_string(n));
    return false;
  }

  if ((frame_align_max_rms_m_ > 0.0 && rms > frame_align_max_rms_m_) ||
      (frame_align_max_error_m_ > 0.0 && max_error > frame_align_max_error_m_))
  {
    std::ostringstream oss;
    oss << "GNSS_FRAME_ALIGN_REJECT reason=residual_too_large yaw_deg=" << yaw_rad * kRadToDeg
        << " tx=" << translation.x()
        << " ty=" << translation.y()
        << " sample_count=" << n
        << " rms=" << rms
        << " max_error=" << max_error;
    logEventThrottled(ros::Time::now().toSec(), "frame_align_reject_residual", 2.0, "WARN", oss.str());
    return false;
  }

  frame_align_yaw_rad_ = yaw_rad;
  frame_align_t_ = translation;
  frame_align_t_.z() = frame_align_translation_.z();
  frame_aligned_ = true;

  std::ostringstream oss;
  oss << "GNSS_FRAME_ALIGN_OK yaw_deg=" << frame_align_yaw_rad_ * kRadToDeg
      << " tx=" << frame_align_t_.x()
      << " ty=" << frame_align_t_.y()
      << " sample_count=" << n
      << " rms=" << rms
      << " max_error=" << max_error;
  logEventThrottled(ros::Time::now().toSec(), "frame_align_ok", 0.0, "INFO", oss.str());
  ROS_INFO("[GNSS] Frame alignment OK: yaw_deg=%.6f tx=%.3f ty=%.3f samples=%d rms=%.3f max_error=%.3f",
           frame_align_yaw_rad_ * kRadToDeg, frame_align_t_.x(), frame_align_t_.y(), n, rms, max_error);
  return true;
}

bool GnssManager::solveFrameAlignment(const std::vector<GnssFrameAlignSample> &solve_samples,
                                      const std::vector<GnssFrameAlignSample> &validation_samples,
                                      double &yaw_rad, V3D &translation,
                                      double &rms, double &max_error) const
{
  if (solve_samples.size() < 2) return false;
  if (sampleMotionExtent(solve_samples) < frame_align_min_motion_m_) return false;

  Eigen::Vector2d mean_enu = Eigen::Vector2d::Zero();
  Eigen::Vector2d mean_world = Eigen::Vector2d::Zero();
  for (const auto &sample : solve_samples)
  {
    mean_enu += sample.enu_position.head<2>();
    mean_world += sample.world_position.head<2>();
  }
  mean_enu /= static_cast<double>(solve_samples.size());
  mean_world /= static_cast<double>(solve_samples.size());

  Eigen::Matrix2d h = Eigen::Matrix2d::Zero();
  for (const auto &sample : solve_samples)
  {
    const Eigen::Vector2d x = sample.enu_position.head<2>() - mean_enu;
    const Eigen::Vector2d y = sample.world_position.head<2>() - mean_world;
    h += x * y.transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix2d> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d r = svd.matrixV() * svd.matrixU().transpose();
  if (r.determinant() < 0.0)
  {
    Eigen::Matrix2d v = svd.matrixV();
    v.col(1) *= -1.0;
    r = v * svd.matrixU().transpose();
  }

  yaw_rad = std::atan2(r(1, 0), r(0, 0));
  const Eigen::Vector2d t = mean_world - r * mean_enu;
  translation << t.x(), t.y(), frame_align_translation_.z();

  const std::vector<GnssFrameAlignSample> &eval_samples =
      validation_samples.empty() ? solve_samples : validation_samples;
  double sq_sum = 0.0;
  max_error = 0.0;
  for (const auto &sample : eval_samples)
  {
    const Eigen::Vector2d predicted = r * sample.enu_position.head<2>() + t;
    const double err = (predicted - sample.world_position.head<2>()).norm();
    sq_sum += err * err;
    max_error = std::max(max_error, err);
  }
  rms = std::sqrt(sq_sum / std::max<size_t>(1, eval_samples.size()));
  return std::isfinite(yaw_rad) && finiteVec3(translation) && std::isfinite(rms) && std::isfinite(max_error);
}

V3D GnssManager::enuToWorld(const V3D &enu) const
{
  const double c = std::cos(frame_align_yaw_rad_);
  const double s = std::sin(frame_align_yaw_rad_);
  V3D world;
  world.x() = c * enu.x() - s * enu.y() + frame_align_t_.x();
  world.y() = s * enu.x() + c * enu.y() + frame_align_t_.y();
  world.z() = enu.z() + frame_align_t_.z();
  return world;
}

GnssUpdateResult GnssManager::rejectResult(const GnssMeasurement &measurement,
                                           const std::string &action,
                                           double time_diff_s,
                                           const V3D &enu,
                                           const V3D &world,
                                           const V3D &pred,
                                           const V3D &residual,
                                           double residual_norm,
                                           double mahalanobis)
{
  GnssUpdateResult result;
  result.action = action;
  result.seq = measurement.seq;
  result.device_state = measurement.state;
  result.source_message = measurement.source_message;
  result.stamp = measurement.stamp;
  result.time_diff_s = time_diff_s;
  result.residual_norm = residual_norm;
  result.mahalanobis_distance = mahalanobis;
  result.sigma_xy = sigma_xy_fixed_m_;
  result.enu_position = enu;
  result.world_position = world;
  result.predicted_position = pred;
  result.residual = residual;
  result.convergence_state = convergenceStateName();
  result.pause_map_update_frames = pause_map_update_frames_;
  result.pause_map_update_min_correction_m = pause_map_update_min_correction_m_;
  logUpdate(result, measurement);
  return result;
}

GnssUpdateResult GnssManager::applyPositionUpdateAt(StatesGroup &state,
                                                   double current_lidar_stamp,
                                                   double lidar_start_stamp)
{
  (void)lidar_start_stamp;
  GnssMeasurement measurement;
  double time_diff_s = 0.0;
  V3D enu = V3D::Zero();
  V3D world = V3D::Zero();
  V3D pred = state.pos_end + state.rot_end * lever_arm_body_to_gnss_;
  V3D residual = V3D::Zero();

  if (!en_)
  {
    GnssUpdateResult result;
    result.action = "disabled";
    return result;
  }

  if (!takeLatestMeasurement(current_lidar_stamp, measurement, time_diff_s))
  {
    bool stale = false;
    double last_stamp = 0.0;
    {
      std::lock_guard<std::mutex> lock(measurement_mutex_);
      stale = have_last_measurement_stamp_;
      last_stamp = last_measurement_stamp_;
    }
    const double reference_stamp = current_lidar_stamp > 0.0 ? current_lidar_stamp : ros::Time::now().toSec();
    if (stale && stale_timeout_s_ > 0.0 && std::fabs(reference_stamp - last_stamp) > stale_timeout_s_)
    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      if (reset_convergence_on_long_stale_ &&
          std::fabs(reference_stamp - last_stamp) > reset_convergence_stale_s_)
      {
        have_first_valid_stamp_ = false;
        was_ready_once_ = false;
        consecutive_fixed_count_ = 0;
        origin_ready_ = origin_mode_ == "manual" && origin_ready_;
        {
          std::lock_guard<std::mutex> measurement_lock(measurement_mutex_);
          have_last_seq_ = false;
          last_seq_ = -1;
          have_last_update_epoch_ = false;
          last_update_epoch_stamp_ = 0.0;
          last_update_epoch_source_.clear();
        }
        if (frame_align_mode_ == "trajectory_2d")
        {
          frame_aligned_ = false;
          frame_align_samples_.clear();
        }
      }
      if (convergence_state_ == ConvergenceState::READY)
      {
        transitionTo(ConvergenceState::DEGRADED, reference_stamp, "GNSS_DATA_STALE");
      }
    }
    GnssUpdateResult result;
    result.action = "no_measurements";
    result.convergence_state = convergenceStateName();
    return result;
  }

  if (stale_timeout_s_ > 0.0 && std::fabs(time_diff_s) > stale_timeout_s_)
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    transitionTo(ConvergenceState::DEGRADED, measurement.stamp, "GNSS_DATA_STALE");
    return rejectResult(measurement, "reject_stale", time_diff_s, enu, world, pred,
                        residual, 0.0, 0.0);
  }
  if (match_threshold_s_ > 0.0 && std::fabs(time_diff_s) > match_threshold_s_)
  {
    return rejectResult(measurement, "reject_stale", time_diff_s, enu, world, pred,
                        residual, 0.0, 0.0);
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    std::string reject_action;
    if (!updateConvergenceAndAlignment(measurement, time_diff_s, state, enu, world, reject_action))
    {
      return rejectResult(measurement, reject_action, time_diff_s, enu, world, pred,
                          residual, 0.0, 0.0);
    }
  }

  if (fixed_only_ && !isFixedSolution(measurement))
  {
    return rejectResult(measurement, "reject_not_fixed", time_diff_s, enu, world, pred,
                        residual, 0.0, 0.0);
  }
  if (!update_en_)
  {
    return rejectResult(measurement, "dry_run", time_diff_s, enu, world, pred,
                        residual, 0.0, 0.0);
  }

  pred = state.pos_end + state.rot_end * lever_arm_body_to_gnss_;
  residual = world - pred;
  const double residual_norm_xy = std::hypot(residual.x(), residual.y());
  if (max_residual_m_ > 0.0 && residual_norm_xy > max_residual_m_)
  {
    return rejectResult(measurement, "reject_large_residual", time_diff_s, enu, world, pred,
                        residual, residual_norm_xy, 0.0);
  }

  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(update_z_ && !update_xy_only_ ? 3 : 2, DIM_STATE);
  Eigen::VectorXd z = Eigen::VectorXd::Zero(h.rows());
  h(0, 3) = 1.0;
  h(1, 4) = 1.0;
  z(0) = residual.x();
  z(1) = residual.y();
  if (h.rows() == 3)
  {
    h(2, 5) = 1.0;
    z(2) = residual.z();
  }
  if (update_orientation_ && !update_xy_only_)
  {
    const Eigen::Matrix3d dpos_dtheta = -state.rot_end * skewSymmetric(lever_arm_body_to_gnss_);
    h.block(0, 0, h.rows(), 3) = dpos_dtheta.block(0, 0, h.rows(), 3);
  }

  MD(DIM_STATE, DIM_STATE) cov_for_gnss = state.cov;
  if (position_cov_floor_m_ > 0.0)
  {
    const double floor_var = position_cov_floor_m_ * position_cov_floor_m_;
    cov_for_gnss(3, 3) = std::max(cov_for_gnss(3, 3), floor_var);
    cov_for_gnss(4, 4) = std::max(cov_for_gnss(4, 4), floor_var);
    if (h.rows() == 3) cov_for_gnss(5, 5) = std::max(cov_for_gnss(5, 5), floor_var);
  }

  Eigen::MatrixXd r = Eigen::MatrixXd::Identity(h.rows(), h.rows()) * (sigma_xy_fixed_m_ * sigma_xy_fixed_m_);
  if (h.rows() == 3) r(2, 2) = sigma_z_fixed_m_ * sigma_z_fixed_m_;

  const Eigen::MatrixXd s = h * cov_for_gnss * h.transpose() + r;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(s);
  if (ldlt.info() != Eigen::Success)
  {
    return rejectResult(measurement, "reject_invalid", time_diff_s, enu, world, pred,
                        residual, residual_norm_xy, 0.0);
  }

  const Eigen::VectorXd s_inv_z = ldlt.solve(z);
  const double mahalanobis = z.dot(s_inv_z);
  if (chi2_gate_2d_ > 0.0 && mahalanobis > chi2_gate_2d_)
  {
    return rejectResult(measurement, "reject_chi2", time_diff_s, enu, world, pred,
                        residual, residual_norm_xy, mahalanobis);
  }

  const Eigen::MatrixXd k = cov_for_gnss * h.transpose() *
                            ldlt.solve(Eigen::MatrixXd::Identity(h.rows(), h.rows()));
  Eigen::VectorXd dx_raw_dynamic = k * z;
  if (dx_raw_dynamic.size() != DIM_STATE || !dx_raw_dynamic.allFinite())
  {
    return rejectResult(measurement, "reject_invalid", time_diff_s, enu, world, pred,
                        residual, residual_norm_xy, mahalanobis);
  }
  if (update_xy_only_)
  {
    for (int i = 0; i < dx_raw_dynamic.size(); ++i)
    {
      if (i != 3 && i != 4) dx_raw_dynamic(i) = 0.0;
    }
  }

  VD(DIM_STATE) dx_raw = VD(DIM_STATE)::Zero();
  dx_raw = dx_raw_dynamic;
  V3D correction_raw = dx_raw.block<3, 1>(3, 0);
  const double correction_raw_norm = std::hypot(correction_raw.x(), correction_raw.y());

  double clamp_ratio = 1.0;
  if (max_update_step_m_ > 0.0 && correction_raw_norm > max_update_step_m_)
  {
    clamp_ratio = max_update_step_m_ / std::max(correction_raw_norm, 1e-9);
  }

  Eigen::MatrixXd k_apply = k * clamp_ratio;
  if (update_xy_only_)
  {
    for (int i = 0; i < k_apply.rows(); ++i)
    {
      if (i != 3 && i != 4) k_apply.row(i).setZero();
    }
  }

  Eigen::VectorXd dx_dynamic = k_apply * z;
  if (dx_dynamic.size() != DIM_STATE || !dx_dynamic.allFinite())
  {
    return rejectResult(measurement, "reject_step_limit", time_diff_s, enu, world, pred,
                        residual, residual_norm_xy, mahalanobis);
  }
  if (update_xy_only_)
  {
    for (int i = 0; i < dx_dynamic.size(); ++i)
    {
      if (i != 3 && i != 4) dx_dynamic(i) = 0.0;
    }
  }

  VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
  dx = dx_dynamic;
  V3D correction_applied = dx.block<3, 1>(3, 0);
  const double correction_applied_norm = std::hypot(correction_applied.x(), correction_applied.y());
  if (max_update_step_m_ > 0.0 && correction_applied_norm > max_update_step_m_ + 1e-6)
  {
    return rejectResult(measurement, "reject_step_limit", time_diff_s, enu, world, pred,
                        residual, residual_norm_xy, mahalanobis);
  }

  state += dx;
  const MD(DIM_STATE, DIM_STATE) i_state = MD(DIM_STATE, DIM_STATE)::Identity();
  const MD(DIM_STATE, DIM_STATE) i_kh = i_state - k_apply * h;
  state.cov = i_kh * cov_for_gnss * i_kh.transpose() + k_apply * r * k_apply.transpose();
  state.cov = 0.5 * (state.cov + state.cov.transpose());
  snapStateForDeterminism(state);
  {
    std::lock_guard<std::mutex> lock(measurement_mutex_);
    have_last_update_epoch_ = true;
    last_update_epoch_stamp_ = measurement.device_time_valid ? measurement.device_stamp : measurement.stamp;
    last_update_epoch_source_ = measurement.source_message;
  }

  GnssUpdateResult result;
  result.state_updated = true;
  result.action = "update_fixed_xy";
  result.seq = measurement.seq;
  result.device_state = measurement.state;
  result.source_message = measurement.source_message;
  result.convergence_state = convergenceStateName();
  result.stamp = measurement.stamp;
  result.time_diff_s = time_diff_s;
  result.residual_norm = residual_norm_xy;
  result.mahalanobis_distance = mahalanobis;
  result.correction_norm = correction_applied_norm;
  result.correction_raw_norm = correction_raw_norm;
  result.correction_applied_norm = correction_applied_norm;
  result.sigma_xy = sigma_xy_fixed_m_;
  result.enu_position = enu;
  result.world_position = world;
  result.predicted_position = pred;
  result.residual = residual;
  result.correction_raw = correction_raw;
  result.correction_applied = correction_applied;
  result.pause_map_update_frames = pause_map_update_frames_;
  result.pause_map_update_min_correction_m = pause_map_update_min_correction_m_;
  result.request_pause_map_insert =
      correction_applied_norm > pause_map_update_min_correction_m_ && pause_map_update_frames_ > 0;
  logUpdate(result, measurement);

  ROS_INFO_THROTTLE(1.0,
                    "[GNSS] update_fixed_xy seq=%d residual=%.3f maha=%.3f raw=%.3f applied=%.3f world=[%.3f %.3f %.3f]",
                    measurement.seq, residual_norm_xy, mahalanobis,
                    correction_raw_norm, correction_applied_norm,
                    world.x(), world.y(), world.z());
  return result;
}

void GnssManager::logRawLine(double stamp, const std::string &line)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!raw_log_file_.is_open()) return;
  raw_log_file_ << std::fixed << std::setprecision(6) << stamp << " raw=\"" << line << "\"\n";
  raw_log_pending_lines_++;
  if (raw_log_pending_lines_ >= log_flush_stride_)
  {
    raw_log_file_.flush();
    raw_log_pending_lines_ = 0;
  }
}

void GnssManager::logParsedMeasurement(const GnssMeasurement &measurement)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!parsed_log_file_.is_open()) return;
  parsed_log_file_ << std::fixed << std::setprecision(9)
                   << measurement.receive_stamp << ' '
                   << measurement.source_message << ' '
                   << static_cast<int>(measurement.checksum_valid) << ' '
                   << static_cast<int>(measurement.device_time_valid) << ' '
                   << measurement.latitude_deg << ' '
                   << measurement.longitude_deg << ' '
                   << measurement.altitude_m << ' '
                   << measurement.raw_position_quality << ' '
                   << solutionTypeName(measurement.solution_type) << ' '
                   << measurement.satellite_count << ' '
                   << measurement.hdop << ' '
                   << measurement.horizontal_std_m << ' '
                   << measurement.vertical_std_m << ' '
                   << measurement.differential_age_s << ' '
                   << static_cast<int>(measurement.valid) << ' '
                   << measurement.reject_reason << " raw=\""
                   << measurement.raw_line << "\"\n";
  parsed_log_pending_lines_++;
  if (parsed_log_pending_lines_ >= log_flush_stride_)
  {
    parsed_log_file_.flush();
    parsed_log_pending_lines_ = 0;
  }
}

void GnssManager::logUpdate(const GnssUpdateResult &result, const GnssMeasurement &measurement)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (!update_log_file_.is_open()) return;
  update_log_file_ << std::fixed << std::setprecision(9)
                   << result.stamp << ' '
                   << result.seq << ' '
                   << result.device_state << ' '
                   << result.convergence_state << ' '
                   << measurement.latitude_deg << ' '
                   << measurement.longitude_deg << ' '
                   << measurement.altitude_m << ' '
                   << result.source_message << ' '
                   << result.enu_position.x() << ' '
                   << result.enu_position.y() << ' '
                   << result.enu_position.z() << ' '
                   << result.world_position.x() << ' '
                   << result.world_position.y() << ' '
                   << result.world_position.z() << ' '
                   << result.predicted_position.x() << ' '
                   << result.predicted_position.y() << ' '
                   << result.residual.x() << ' '
                   << result.residual.y() << ' '
                   << result.residual_norm << ' '
                   << result.sigma_xy << ' '
                   << result.mahalanobis_distance << ' '
                   << result.time_diff_s << ' '
                   << result.correction_raw_norm << ' '
                   << result.correction_applied_norm << ' '
                   << result.action << '\n';
  update_log_pending_lines_++;
  if (update_log_pending_lines_ >= log_flush_stride_)
  {
    update_log_file_.flush();
    update_log_pending_lines_ = 0;
  }
}

void GnssManager::logEventThrottled(double stamp, const std::string &key, double period_s,
                                    const std::string &level, const std::string &message)
{
  std::lock_guard<std::mutex> lock(log_mutex_);
  double &last_stamp = event_log_last_stamp_[key];
  if (period_s > 0.0 && last_stamp > 0.0 && stamp - last_stamp < period_s) return;
  last_stamp = stamp;
  if (update_log_file_.is_open())
  {
    update_log_file_ << std::fixed << std::setprecision(6)
                     << stamp << " event " << level << ' ' << message << '\n';
    update_log_file_.flush();
  }
}

void GnssManager::transitionTo(ConvergenceState state, double stamp, const std::string &event)
{
  if (convergence_state_ == state && event != "GNSS_FIXED_CONFIRMED") return;
  convergence_state_ = state;
  logEventThrottled(stamp, event, 1.0, "INFO", event);
  if (event == "GNSS_READY")
  {
    ROS_INFO_THROTTLE(1.0, "[GNSS] GNSS_READY");
  }
  else if (event == "GNSS_FIXED_LOST" || event == "GNSS_DATA_STALE")
  {
    ROS_WARN_THROTTLE(1.0, "[GNSS] %s", event.c_str());
  }
}

std::string GnssManager::convergenceStateName() const
{
  switch (convergence_state_)
  {
    case ConvergenceState::DISABLED: return "DISABLED";
    case ConvergenceState::SERIAL_OPENED: return "SERIAL_OPENED";
    case ConvergenceState::WAIT_VALID_DATA: return "WAIT_VALID_DATA";
    case ConvergenceState::WARMING_UP: return "WARMING_UP";
    case ConvergenceState::WAIT_FIXED: return "WAIT_FIXED";
    case ConvergenceState::ALIGNING: return "ALIGNING";
    case ConvergenceState::READY: return "READY";
    case ConvergenceState::DEGRADED: return "DEGRADED";
  }
  return "UNKNOWN";
}
