#pragma once
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

// An isolated qualification ceiling, never selected by the serving adapter.
inline int publisher_probe_bitrate() {
  const char* guard=std::getenv("THA_CLAIRE_RATE_PROBE");
  const char* value=std::getenv("THA_CLAIRE_PROBE_BITRATE_BPS");
  if(!guard&&!value)return 0;
  if(!guard||std::strcmp(guard,"1")!=0||!value)throw std::runtime_error("explicit rate-probe guard and ceiling required");
  for(int allowed:{300000,600000,1200000,2400000})if(std::to_string(allowed)==value)return allowed;
  throw std::runtime_error("unsupported rate-probe ceiling");
}

inline bool publisher_probe_scheduled() {
  const char* schedule=std::getenv("THA_CLAIRE_RATE_PROBE_SCHEDULE");
  if(!schedule)return false;
  if(std::strcmp(schedule,"down-up")!=0||publisher_probe_bitrate()!=1200000)
    throw std::runtime_error("explicit down-up probe with 1200000 initial ceiling required");
  return true;
}

inline int publisher_probe_ceiling(int initial,bool scheduled,std::uint64_t source_sample) {
  return scheduled&&source_sample>=144000&&source_sample<288000?300000:initial;
}
