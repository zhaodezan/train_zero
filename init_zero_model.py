from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
import torch, json, os

model_id = "./tokenizer_zh_en_math_code_16k"
save_path = "./empty_qwen35_custom_init"
os.makedirs(save_path, exist_ok=True)

# 1. 加载配置并构建空模型（用多模态类，不是 CausalLM）
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_config(config)

# 2. 手动初始化（可选）
for module in model.modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

# 3. 保存模型（config.json 会自动包含正确的 architectures）
model.save_pretrained(save_path)

# 4. 保存 processor（包含 tokenizer + 视觉预处理配置）
processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(save_path)

# 5. 不需要手动改 architectures，save_pretrained 已写入正确值
print("已保存：", os.listdir(save_path))
