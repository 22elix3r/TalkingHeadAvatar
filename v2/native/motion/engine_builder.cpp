// Build only the inspected Claire regression shape profile; no external engines.
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
void require(bool valid,const char* message) {if(!valid) throw std::runtime_error(message);}
struct Logger final: nvinfer1::ILogger {
  void log(Severity severity,const char* message) noexcept override {
    if(severity<=Severity::kWARNING) std::cerr<<"TensorRT: "<<message<<'\n';
  }
};
}
int main(int argc,char** argv) {
  try {
    require(argc==3,"usage: tha_claire_engine pinned-network.onnx new-private-network.trt");
    const char* inherited=std::getenv("THA_DEVICE_LEASE_FD");
    require(inherited && fcntl(std::stoi(inherited),F_GETFD)>=0,"inherited admitted device lease required");
    require(getInferLibVersion()==NV_TENSORRT_VERSION,"TensorRT header/runtime mismatch");
    const auto start=std::chrono::steady_clock::now();Logger logger;
    auto builder=std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    require(bool(builder),"TensorRT builder unavailable");require(builder->setMaxThreads(4),"bounded builder threads required");
    auto network=std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    require(bool(network),"network allocation failed");
    auto parser=std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network,logger));
    require(parser && parser->parseFromFile(argv[1],static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)),"pinned ONNX parse failed");
    require(network->getNbInputs()==2 && network->getNbOutputs()==1,"Claire network tensor inventory mismatch");
    auto* profile=builder->createOptimizationProfile(); // Owned by IBuilder.
    auto config=std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    require(profile && config,"builder profile/config allocation failed");
    for(int index=0;index<2;++index) {
      auto* input=network->getInput(index);const std::string name=input->getName();const auto dims=input->getDimensions();
      require((name=="input" || name=="emotion") && dims.nbDims==3 && dims.d[1]==1 && dims.d[2]==(name=="input"?8320:26),"unexpected Claire input shape");
      const nvinfer1::Dims3 shape{1,1,name=="input"?8320:26};
      for(auto selector:{nvinfer1::OptProfileSelector::kMIN,nvinfer1::OptProfileSelector::kOPT,nvinfer1::OptProfileSelector::kMAX})
        require(profile->setDimensions(name.c_str(),selector,shape),"fixed N=1 profile rejected");
    }
    require(profile->isValid() && config->addOptimizationProfile(profile)==0,"N=1 optimization profile invalid");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,256ULL*1024*1024);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY,48*1024);
    config->setMaxAuxStreams(0);config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    auto plan=std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network,*config));
    require(plan && plan->size()>0 && plan->size()<1024ULL*1024*1024,"bounded engine build failed");
    const int output=open(argv[2],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    require(output>=0,"new private engine output required");
    std::size_t written=0;const auto* data=static_cast<const char*>(plan->data());
    while(written<plan->size()) {const auto count=write(output,data+written,plan->size()-written);if(count<=0) {close(output);throw std::runtime_error("engine write failed");}written+=count;}
    require(fsync(output)==0,"engine persistence failed");close(output);
    std::cout<<"{\"engine_bytes\":"<<plan->size()<<",\"fixed_batch\":1,\"tf32\":false,\"workspace_limit_mib\":256,\"tactic_shared_memory_bytes\":49152,\"elapsed_ms\":"
      <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<"Claire engine build: "<<error.what()<<'\n';return 1;}
}
