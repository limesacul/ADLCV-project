from safetensors.torch import load_file

lora_weights_path = "/dtu/blackhole/07/203495/ADLCV-project/lora_weights/adapter_model.safetensors"
lora_weights = load_file(lora_weights_path)
print(lora_weights.keys())