# AI Model Fine-Tuning & Inference

This repository contains tools for fine-tuning the TinyLlama model on a custom dataset, merging the trained LoRA adapters, and running inference with a CLI chat interface.

## 📂 Project Structure

- **`dataset.json`**: The training dataset in JSON format containing instruction-response pairs.
- **`finetune.py`**: Script to fine-tune the base TinyLlama model using LoRA (Low-Rank Adaptation).
- **`merge_checkpoint.py`**: Script to merge the fine-tuned LoRA adapter (specifically checkpoint-19) with the base model.
- **`test.py`**: A command-line interface (CLI) to chat with the merged, fine-tuned model.
- **`requirement.txt`**: List of Python dependencies required to run the project.

## 🚀 Setup & Installation

### 1. Install Dependencies
Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
pip install -r requirement.txt
```

*Note: You may need to install PyTorch separately depending on your system (CUDA/CPU) if the default installation doesn't match your hardware.*

### 2. Download Base Model
The scripts assume the base **TinyLlama** model is located in a local directory named `Tinyllama`.
1. Download a generic TinyLlama model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) from Hugging Face.
2. Place the model files inside a folder named `Tinyllama` in the root of this project.

## 📊 Dataset Format

The `dataset.json` file follows a standard Instruction-Input-Output format suitable for instruction tuning.

```json
[
    {
        "instruction": "What is Python?",
        "input": "",
        "output": "Python is a high-level, interpreted programming language known for its readability."
    },
    ...
]
```
- **instruction**: The user's query or task.
- **input**: Optional context or additional input (can be empty).
- **output**: The desired model response.

## 🛠️ Usage Guide

### Step 1: Fine-Tuning
Run the `finetune.py` script to start training. 
*Note: The script is currently configured for CPU training (`device_map="cpu"`). Update the script if you have a GPU.*

```bash
python finetune.py
```
- **Output**: Checkpoints will be saved in the `tiny-finetuned` directory.
- **Configuration**: You can modify `finetune.py` to adjust `num_train_epochs`, `batch_size`, or LoRA parameters.

### Step 2: Merging the Model
Once training is complete, you need to merge the adapter weights with the base model to create a standalone model for inference.

The script `merge_checkpoint_19.py` is hardcoded to look for **checkpoint-19**. Check your `tiny-finetuned` folder for the actual checkpoint number and update the script if necessary.

```bash
python merge_checkpoint.py
```
- **Output**: The merged model will be saved to `tiny-finetuned-merged`.

### Step 3: Inference (Chat)
Run the `test.py` script to chat with your newly trained model.

```bash
python test.py
```
This launches an interactive CLI chat session. Type `exit` or `quit` to end the session.

## ⚠️ Important Notes

- **Model Paths**: ensure the folder `Tinyllama` exists before running the scripts.
- **Checkpoints**: The merge script specifically targets `checkpoint-19`. If your training runs for fewer or more steps, please update the path in `merge_checkpoint_19.py` (line 9).
- **Hardware**: Fine-tuning on CPU is slow. For faster results, ensure you have CUDA installed and modify `finetune.py` to remove `device_map={"": "cpu"}` or set it to `"auto"`.



