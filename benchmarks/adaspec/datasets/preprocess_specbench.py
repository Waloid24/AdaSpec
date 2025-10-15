import copy

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
import jsonlines
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import random
random.seed(0)

def convert_question_to_input_ids(question, conv, tokenizer):
    qs = question["turns"][0]
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    conv.stop_str = "</s>"
    prompt = conv.get_prompt()
    # inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    inputs = tokenizer(prompt)
    return prompt, inputs.input_ids

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
dataset = load_questions("./origin-spec-bench.jsonl", 0, 480)
output_len = 1024

sampling_params = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=False)

llm = LLM(
    model="lmsys/vicuna-7b-v1.5",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    seed=0,
)

with jsonlines.open('./spec-bench.jsonl', mode='w') as writer:
    for question in dataset:
        conv = get_conversation_template("vicuna_v1.1")
        prompt, input_ids = convert_question_to_input_ids(question, conv, tokenizer)
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        new_question = copy.copy(question)
        new_question['output_len'] = len(outputs[0].outputs[0].token_ids)
        writer.write(new_question)

output_lens = []
output_dict = {
    "writing": [],
    "roleplay": [],
    "reasoning": [],
    "math": [],
    "coding": [],
    "extraction": [],
    "stem": [],
    "humanities": [],
    "translation": [],
    "summarization": [],
    "qa": [],
    "math_reasoning": [],
    "rag": []
}
dataset = load_questions("./spec-bench.jsonl", 0, 480)
for question in dataset:
    output_lens.append(question['output_len'])
    output_dict[question['category']].append(question['output_len'])
print(np.mean(output_lens))
for key, value in output_dict.items():
    print(key, np.mean(value))