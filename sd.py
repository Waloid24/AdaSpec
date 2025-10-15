from vllm import LLM, SamplingParams
import time
import os
import random
from fastchat.llm_judge.common import load_questions

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 8
output_len = 128

# Load the dataset.
dataset = load_questions("./benchmarks/adaspec/datasets/spec-bench.jsonl", 0, 480)

# Shuffle the dataset.
random.shuffle(dataset)

prompts = []
for i in range(batch_size):
  question = dataset[random.randint(0, 479)]
  prompts.append(question['turns'][0])

sampling_params = SamplingParams(temperature=0, max_tokens=output_len,ignore_eos=True)


llm = LLM(
    model="lmsys/vicuna-7b-v1.5",
    tensor_parallel_size=1,
    speculative_model="double7/vicuna-68m",
    num_speculative_tokens=8,
    enable_chunked_prefill=False,
    use_v2_block_manager=True,
    disable_async_output_proc=True,
    gpu_memory_utilization=0.9,
    seed=0,
    dynamic_spec=False,
    ssd=True,
    rsd=True,
    dsd=False,
)

start = time.time()
outputs = llm.generate(prompts, sampling_params=sampling_params)
end = time.time()

batch_time = end - start
del llm

# Throughput and time
sum_outputs = 0

for output in outputs:
    sum_outputs += len(output)

print(f"Throughput: {sum_outputs / batch_time:.2f} tokens/s")
print(f"Batch expansion time: {batch_time:.2f}s")
