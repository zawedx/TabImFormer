from transformers import AutoModelForCausalLM, AutoTokenizer, models

model_name = "dev/qwen"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Hi."
messages = [
    {"role": "system", "content": "You are qwen."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
s = models.qwen2.modeling_qwen2.Qwen2ForCausalLM
print(type(model))
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=400,
    temperature=1.0,
    top_k=50,
    top_p=0.9
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)