from transformers import AutoTokenizer, AutoModelForCausalLM,Trainer,TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

#-------------------------------------------------
#1.Model & Tokenizer
#-------------------------------------------------
model_name ="Tinyllama" #base model folder
print(( " 😊 Loading Model & Tokenizer........."))
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map={"": "cpu"})#set device map if avalible gpu then use gpu else cpu


#-------------------------------------------------
# 2. Apply LoRA configuration
#-------------------------------------------------
print(" 😊 Applying LoRA configuration.........")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,#set r
    lora_alpha=16,#set lora alpha
    lora_dropout=0.05,#set lora dropout
    bias="none",#set bias
    target_modules=["q_proj", "v_proj"],#set target modules
)
model = get_peft_model(model, peft_config)

#-------------------------------------------------
# 3. Dataset Loading
#-------------------------------------------------
print(" 😊 Loading Dataset.........")
dataset = load_dataset("json", data_files="dataset.json")["train"] #dataset folder

def format_and_tokenize(examples):
    text = f"instruction: {examples['instruction']}\ninput: {examples['input']}\noutput: {examples['output']}"
    tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

#-------------------------------------------------
# 4. Training Arguments
#-------------------------------------------------
print(" 😊 Setting Training Arguments.........")
training_args = TrainingArguments(
    output_dir="tiny-finetuned", #output folder
    overwrite_output_dir=True,#set overwrite output dir
    num_train_epochs=1, #set number or epochs
    per_device_train_batch_size=1,#set per device train batch size
    gradient_accumulation_steps=8,#set gradient accumulation steps
    learning_rate=5e-4, #set learning rate
    weight_decay=0.01,#set weight decay
    logging_steps=10,#set logging steps
    save_strategy="epoch",#set save strategy
    fp16=False, #if True, use fp16 training cpu only 
    bf16=False, #if True, use bf16 training cpu only
    logging_dir="logs",#set logging dir
)

#------------------------------------------------
# 5. Start Training
#------------------------------------------------
print(" 😊 Starting Training.........(this will take time on CPU)")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()


#------------------------------------------------
# 6. Save fine-tuned model
#------------------------------------------------
print(" 🛟😍 Saving fine-tuned model.........")
model.save_pretrained(" ./tinyllam-finetuned")#set save model
tokenizer.save_pretrained(" ./tinyllam-finetuned")#set save tokenizer

print(" 🛟😍 Training completed & Model saved successfully in ./tinyllam-finetuned")
















    
    


