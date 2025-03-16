from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'samitizerxu/Qwen2.5-R1-Distill-GRPO-h'
quant_path = 'Qwen2.5-R1-Distill-GRPO-h-awq'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_cosmopedia():
    data = load_dataset('open-r1/OpenR1-Math-220k', "extended")
    data = data.filter(lambda x: len(x["problem"])+len(x["solution"]) >= 3000)
    all_rows = [row["problem"]+row["solution"] for row in data["train"]]
    print(len(all_rows))
    return all_rows

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_cosmopedia(),
    # n_parallel_calib_samples=32,
    max_calib_samples=128,
    max_calib_seq_len=1024
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')

