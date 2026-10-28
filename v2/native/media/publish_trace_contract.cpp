// Native ABI guards without a room, engine, model or GPU allocation.
#include "publish_trace.h"
#include <functional>
#include <string>

static void rejected(const std::function<void()>& action,const std::string& expected) {
  try{action();}catch(const std::runtime_error& error){
    if(std::string(error.what()).find(expected)!=std::string::npos)return;
    throw;
  }
  throw std::runtime_error("publisher trace guard accepted an invalid boundary");
}

int main(int argc,char** argv) {
  try{
    if(argc!=2)throw std::runtime_error("one locally verified trace FFI path required");
    unsetenv("THA_CLAIRE_PUBLISH_TRACE");unsetenv("THA_CLAIRE_TRACE_FFI_PATH");
    {PublishStageTrace disabled(0);disabled.finish();}
    setenv("THA_CLAIRE_PUBLISH_TRACE","1",1);
    rejected([]{PublishStageTrace unavailable(0);},"ABI unavailable");
    void* library=dlopen(argv[1],RTLD_NOW|RTLD_GLOBAL);
    if(!library)throw std::runtime_error("locally built trace FFI load failed");
    setenv("THA_CLAIRE_TRACE_FFI_PATH","wrong-local-library",1);
    rejected([]{PublishStageTrace wrong(0);},"identity mismatch");
    setenv("THA_CLAIRE_TRACE_FFI_PATH",argv[1],1);
    rejected([]{PublishStageTrace missing_track(0);},"attachment failed");
    dlclose(library);
    std::cout<<"publisher trace ABI guards passed\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
