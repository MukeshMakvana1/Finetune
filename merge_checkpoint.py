
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def merge_model():
    base_model_path = os.path.abspath("./Tinyllama")#base model folder
    adapter_path = os.path.abspath("./tiny-finetuned/checkpoint-19")#adapter folder
    output_path = os.path.abspath("./tiny-finetuned-merged")#output folder

    print(f"Base model: {base_model_path}")#set base model path
    print(f"Adapter: {adapter_path}")#set adapter path
    print(f"Output: {output_path}")#set output path

    device = "cpu" #set device map if avalible gpu then use gpu else cpu
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device,
        low_cpu_mem_usage=True,#set low cpu mem usage
        torch_dtype=torch.float16#set torch dtype
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device,
        torch_dtype=torch.float16
    )

    print("Merging...")
    merged_model = model.merge_and_unload()

    print("Saving merged model...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    print("Done!")

if __name__ == "__main__":
    merge_model()
