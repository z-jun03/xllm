#pragma once

#include "chat_service_impl.h"
#include "completion_service_impl.h"
#include "embedding_service_impl.h"
#include "models_service_impl.h"
#include "xllm_service.pb.h"

namespace xllm {

class APIService : public proto::XllmAPIService {
 public:
  APIService(Master* master,
             const std::vector<std::string>& model_names,
             const std::vector<std::string>& model_versions);
  ~APIService() = default;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::CompletionRequest* request,
                   proto::CompletionResponse* response,
                   ::google::protobuf::Closure* done) override;

  void CompletionsHttp(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::ChatRequest* request,
                       proto::ChatResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ChatCompletionsHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::EmbeddingRequest* request,
                  proto::EmbeddingResponse* response,
                  ::google::protobuf::Closure* done) override;

  void EmbeddingsHttp(::google::protobuf::RpcController* controller,
                      const proto::HttpRequest* request,
                      proto::HttpResponse* response,
                      ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::ModelListRequest* request,
              proto::ModelListResponse* response,
              ::google::protobuf::Closure* done) override;

  void ModelsHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void GetCacheInfo(::google::protobuf::RpcController* controller,
                    const proto::HttpRequest* request,
                    proto::HttpResponse* response,
                    ::google::protobuf::Closure* done) override;

  void LinkCluster(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void UnlinkCluster(::google::protobuf::RpcController* controller,
                     const proto::HttpRequest* request,
                     proto::HttpResponse* response,
                     ::google::protobuf::Closure* done) override;

  void ModelVersionsHttp(::google::protobuf::RpcController* controller,
                         const proto::HttpRequest* request,
                         proto::HttpResponse* response,
                         ::google::protobuf::Closure* done) override;

 private:
  Master* master_;

  std::unique_ptr<CompletionServiceImpl> completion_service_impl_;
  std::unique_ptr<ChatServiceImpl> chat_service_impl_;
  std::unique_ptr<MMChatServiceImpl> mm_chat_service_impl_;
  std::unique_ptr<EmbeddingServiceImpl> embedding_service_impl_;
  std::unique_ptr<ModelsServiceImpl> models_service_impl_;
};

}  // namespace xllm
