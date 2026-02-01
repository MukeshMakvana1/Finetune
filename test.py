from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import torch

# Try loading the full model in `Tinyllama`. If you intended to use an adapter
# from `tiny-finetuned`, you'd load a base model then apply the adapter via PEFT.
# Path to the standalone merged model
model_path = "./tiny-finetuned-merged"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}. Loading model...")

try:
    # Load the standalone merged model
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map={"": "cpu"}, 
            low_cpu_mem_usage=True
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)

except Exception as e:
    print("Failed to load local model:", e)
    print("Falling back to a small model 'gpt2' for a quick test.")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# Chat Loop
def chat():
    print("\n" + "="*50)
    print("Kaliyo AI CLI Chat (Type 'quit' or 'exit' to stop)")
    print("="*50 + "\n")
    
    # Store history
    history = []
    
    # System prompt
    system_prompt = "You are Kaliyo AI, a helpful and knowledgeable assistant. You answer questions thoughtfully."
    
    while True:
        try:
            user_input = input("\n\033[1;32mUser:\033[0m ") # Green for User
            if user_input.strip().lower() in ['quit', 'exit']:
                print("\n\033[1;33mGoodbye!\033[0m")
                break
            
            # Construct prompt with simplified history (last 5 turns to fit context)
            # Format: System -> User -> Assistant -> User -> Assistant...
            full_prompt = f"{system_prompt}\n\n"
            
            # Add recent history
            for sender, msg in history[-5:]:
                full_prompt += f"{sender}: {msg}\n"
            
            # Add current input
            full_prompt += f"User: {user_input}\nAssistant:"
            
            # Generate
            # We set max_new_tokens to avoid generating too much, and handle stopping criteria loosely via string checking
            output = pipe(
                full_prompt, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9,
                return_full_text=False # Only return the generated part
            )
            
            generated_text = output[0]["generated_text"].strip()
            
            # Simple cleanup if model hallucinates subsequent turns
            if "User:" in generated_text:
                generated_text = generated_text.split("User:")[0].strip()
            
            print(f"\033[1;36mKaliyo AI:\033[0m {generated_text}") # Cyan for AI
            
            # Update history
            history.append(("User", user_input))
            history.append(("Assistant", generated_text))
            
        except KeyboardInterrupt:
            print("\n\n\033[1;33mGoodbye!\033[0m")
            break
        except Exception as e:
            print(f"\n\033[1;31mError: {e}\033[0m")

if __name__ == "__main__":
    chat()

