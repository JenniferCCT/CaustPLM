from modelscope import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "AI-ModelScope/gpt2",
    trust_remote_code=True
)

prompt_text = (
    "<|start_prompt|>"
    "The dataset is a real-world traffic sensor dataset collected from the PEMS08 road network. "
    "Each row represents a traffic sensor, and each sensor has a time series length of 12. "
    "Some values in the dataset are missing, and the missing values are marked as 0. "
    "#Instruction:# "
    "The task is to learn temporal dependencies within each sensor and spatial correlations across sensors "
    "to infer and fill in the missing traffic values. "
    "Finally, only return the completed dataset after filling in the missing values."
)

tok = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=False)

print("Prompt length:", tok["input_ids"].shape[1])