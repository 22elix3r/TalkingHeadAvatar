// Bounded, opt-in SDK-stage metadata. The C ABI exists only in the isolated
// locally built trace overlay; normal serving does not enable or require it.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

class PublishStageTrace {
  struct Row {
    uint32_t stage,has_frame_id,frame_id,reserved;
    uint64_t user_timestamp_us,observed_unix_us;
  };
  static_assert(sizeof(Row)==32,"publisher trace ABI layout changed");
  using Abi=uint32_t(*)();
  using Begin=uint64_t(*)(uint64_t);
  using End=size_t(*)(uint64_t,Row*,size_t,uint64_t*);
  static constexpr size_t maximum=16384;
  End end_=nullptr;
  uint64_t handle_=0;
 public:
  explicit PublishStageTrace(uint64_t track) {
    const char* enabled=std::getenv("THA_CLAIRE_PUBLISH_TRACE");
    if(!enabled||std::strcmp(enabled,"1")!=0)return;
    auto abi=reinterpret_cast<Abi>(dlsym(RTLD_DEFAULT,"tha_lk_publish_trace_abi"));
    auto begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"tha_lk_publish_trace_begin"));
    end_=reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"tha_lk_publish_trace_end"));
    if(!abi||abi()!=1||!begin||!end_)throw std::runtime_error("qualified publisher trace ABI unavailable");
    const char* expected=std::getenv("THA_CLAIRE_TRACE_FFI_PATH");Dl_info loaded{};
    if(!expected||!dladdr(reinterpret_cast<void*>(abi),&loaded)||!loaded.dli_fname||std::strcmp(expected,loaded.dli_fname)!=0)
      throw std::runtime_error("publisher trace FFI identity mismatch");
    handle_=begin(track);
    if(!handle_)throw std::runtime_error("publisher trace attachment failed");
  }
  PublishStageTrace(const PublishStageTrace&)=delete;
  PublishStageTrace& operator=(const PublishStageTrace&)=delete;
  void finish() {
    if(!handle_)return;
    std::vector<Row> rows(maximum);uint64_t overflow=0;
    const auto count=end_(handle_,rows.data(),rows.size(),&overflow);handle_=0;
    if(count==std::numeric_limits<size_t>::max()||count>maximum)throw std::runtime_error("publisher trace snapshot failed");
    std::ostringstream output;
    output<<"THA_PUBLISH_TRACE_BOUND "<<count<<' '<<overflow<<'\n';
    for(size_t index=0;index<count;++index){
      const auto& row=rows[index];
      if(row.stage<1||row.stage>3||row.has_frame_id>1||row.reserved)throw std::runtime_error("publisher trace row invalid");
      output<<"THA_PUBLISH_TRACE "<<row.stage<<' '<<row.has_frame_id<<' '<<row.frame_id<<' '
        <<row.user_timestamp_us<<' '<<row.observed_unix_us<<'\n';
    }
    std::cerr<<output.str();
    if(overflow)throw std::runtime_error("publisher trace overflow; observation is incomplete");
  }
  ~PublishStageTrace() {
    try{finish();}catch(...){std::cerr<<"THA_PUBLISH_TRACE_INCOMPLETE\n";}
  }
};
