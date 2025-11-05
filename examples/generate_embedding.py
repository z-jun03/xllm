# python examples/generate_embedding.py --model='/path/models/Qwen3-8B' --devices='npu:0'
# python generate_embedding.py --model='/path/models/Qwen3-8B' --devices='npu:0,npu:1'

from xllm import ArgumentParser, Embedding, RequestParams

# Create an EmbeddingLM.
parser = ArgumentParser()
emb = Embedding(**vars(parser.parse_args()))

# Create a reqeust params, include sampling params
request_params = RequestParams()
request_params.is_embeddings = True
request_params.max_tokens = 1

inputs = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = emb.embedding(inputs, request_params, True)

# Print the outputs.
for i, output in enumerate(outputs):
    input_str = output.prompt
    generated_embedding = output.outputs[0].embeddings
    print(f"Input: {input_str!r}, Generated embedding: {generated_embedding!r}")

emb.finish()

