// Qualification settings stay opt-in, bounded and independent of GPU access.
#include "probe_rate.h"
#include <iostream>

static void rejected(const char* guard,const char* value) {
  if(guard)setenv("THA_CLAIRE_RATE_PROBE",guard,1);else unsetenv("THA_CLAIRE_RATE_PROBE");
  if(value)setenv("THA_CLAIRE_PROBE_BITRATE_BPS",value,1);else unsetenv("THA_CLAIRE_PROBE_BITRATE_BPS");
  try{publisher_probe_bitrate();}catch(const std::runtime_error&){return;}
  throw std::runtime_error("invalid rate-probe selection accepted");
}

int main() {
  try{
    unsetenv("THA_CLAIRE_RATE_PROBE");unsetenv("THA_CLAIRE_PROBE_BITRATE_BPS");
    if(publisher_probe_bitrate()!=0)throw std::runtime_error("rate probe enabled by default");
    rejected("1",nullptr);rejected(nullptr,"600000");rejected("0","600000");rejected("true","600000");
    for(const char* value:{"","0","100000","600000 ","0600000","-600000","600000x","999999999999999999999"})rejected("1",value);
    setenv("THA_CLAIRE_RATE_PROBE","1",1);
    for(int expected:{300000,600000,1200000,2400000}){
      setenv("THA_CLAIRE_PROBE_BITRATE_BPS",std::to_string(expected).c_str(),1);
      if(publisher_probe_bitrate()!=expected)throw std::runtime_error("rate-probe selection mismatch");
    }
    std::cout<<"bounded rate-probe contracts passed\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
