#include "word_embedding.h"

#include "atb_word_embedding_impl.h"

namespace xllm::hf {

std::shared_ptr<AtbEmbeddingImpl> create_word_embedding_layer(
    const Context& context) {
  return std::make_shared<AtbWordEmbeddingImpl>(context);
}

AtbWordEmbedding::AtbWordEmbedding(const Context& context)
    : ModuleHolder(create_word_embedding_layer(context)) {}

}  // namespace xllm::hf
