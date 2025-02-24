import torch, os, pathlib
import numpy as np
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

from transformers import AutoTokenizer, AutoModelForCausalLM

from llm.utils.evalutate import evaluate, print_size_of_model

torch.backends.quantized.engine = 'qnnpack'
# Preprocess the validation dataset
def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(
        examples["question1"], 
        examples["question2"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )
    

def main(model_name = "gpt2", quantized_model_path = "../app/assets/gpt2_quantized.pth"):
    # Load model
    torch.backends.quantized.engine = 'qnnpack'
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Apply dynamic quantization (Reduces size and improves CPU inference speed)
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # quantize model

    dir_path = "/".join(quantized_model_path.split("/")[:-1])
    os.makedirs(dir_path)
    torch.save(quantized_model.state_dict(), quantized_model_path)

    print(f"Quantized model saved at: {quantized_model_path}")
    print_size_of_model(model)
    print_size_of_model(quantized_model)
    
    # Load the MRPC dataset
    login(token=os.environ['HF_TOKEN'])
    dataset = DatasetDict()
    dataset['train'] = load_dataset("DT4LM/gpt2_mrpc_leap", split='train[:80%]')
    dataset['test'] = load_dataset("DT4LM/gpt2_mrpc_leap", split='train[80%:90%]')
    dataset['validation'] = load_dataset("DT4LM/gpt2_mrpc_leap", split='train[90%:]')
    
    eval_dataset = dataset["validation"]
    encoded_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    encoded_eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Run evaluation
    evaluate(quantized_model, encoded_eval_dataset)

