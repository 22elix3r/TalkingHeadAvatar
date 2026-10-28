#pragma once
#include <cstdlib>
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
