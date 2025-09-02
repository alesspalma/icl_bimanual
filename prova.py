import os
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 0

# fix all seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For PyTorch 1.8+
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True # Note that this Deterministic mode can have a performance impact
torch.use_deterministic_algorithms(True)

name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype="auto",
    device_map="auto",
    # max_memory={0: "12GB"}
)
tokenizer = AutoTokenizer.from_pretrained(name)
for param in model.parameters():
    param.requires_grad = False # no fine-tuning

def huggingface_call(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


messages = [
            {"role": "system", "content": "you are a general assistant"},
            {"role": "user", "content": "say 15 random words."}
        ]

print(huggingface_call(model, tokenizer, messages))