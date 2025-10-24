# python examples/generate.py --model='/path/models/Qwen2-7B-Instruct' --devices='npu:0'
# python generate.py --model='/path/models/Qwen2-7B-Instruct' --devices='npu:0,npu:1'

from xllm import ArgumentParser, LLM, RequestParams

# Create an LLM.
parser = ArgumentParser()
llm = LLM(**vars(parser.parse_args()))

# Create a reqeust params, include sampling params
request_params = RequestParams()
request_params.temperature = 0.8
request_params.top_p = 0.95
request_params.max_tokens = 10

# Generate texts from the prompts. The output is a list of RequestOutput
# objects that contain the prompt, generated text, and other information.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, request_params, True)

# Print the outputs.
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

llm.finish()

