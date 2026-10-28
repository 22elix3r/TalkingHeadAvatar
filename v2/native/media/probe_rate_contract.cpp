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
    unsetenv("THA_CLAIRE_RATE_PROBE");unsetenv("THA_CLAIRE_PROBE_BITRATE_BPS");unsetenv("THA_CLAIRE_RATE_PROBE_SCHEDULE");
    if(publisher_probe_scheduled())throw std::runtime_error("rate schedule enabled by default");
    if(publisher_probe_bitrate()!=0)throw std::runtime_error("rate probe enabled by default");
    rejected("1",nullptr);rejected(nullptr,"600000");rejected("0","600000");rejected("true","600000");
    for(const char* value:{"","0","100000","600000 ","0600000","-600000","600000x","999999999999999999999"})rejected("1",value);
    setenv("THA_CLAIRE_RATE_PROBE","1",1);
    for(int expected:{300000,600000,1200000,2400000}){
      setenv("THA_CLAIRE_PROBE_BITRATE_BPS",std::to_string(expected).c_str(),1);
      if(publisher_probe_bitrate()!=expected)throw std::runtime_error("rate-probe selection mismatch");
    }
    setenv("THA_CLAIRE_RATE_PROBE_SCHEDULE","down-up",1);
    bool failed=false;try{publisher_probe_scheduled();}catch(const std::runtime_error&){failed=true;}
    if(!failed)throw std::runtime_error("schedule accepted a different initial ceiling");
    setenv("THA_CLAIRE_PROBE_BITRATE_BPS","1200000",1);
    if(!publisher_probe_scheduled())throw std::runtime_error("explicit schedule rejected");
    for(const auto sample:{0ULL,143999ULL,288000ULL,600000ULL})
      if(publisher_probe_ceiling(1200000,true,sample)!=1200000)throw std::runtime_error("wrong high-rate interval");
    for(const auto sample:{144000ULL,144001ULL,287999ULL})
      if(publisher_probe_ceiling(1200000,true,sample)!=300000)throw std::runtime_error("wrong low-rate interval");
    if(publisher_probe_ceiling(600000,false,144000)!=600000)throw std::runtime_error("fixed ceiling changed");
    setenv("THA_CLAIRE_RATE_PROBE_SCHEDULE","unknown",1);failed=false;
    try{publisher_probe_scheduled();}catch(const std::runtime_error&){failed=true;}
    if(!failed)throw std::runtime_error("unknown schedule accepted");
    std::cout<<"bounded rate-probe contracts passed\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
