set -e

# 1. 训练 tokenizer（已有结果时不覆盖）
if [ ! -d "tokenizer_zh_en_math_code_16k" ]; then
    python train_tokenizer.py
fi
# 2. 查看 tokenizer 编码与解码示例
python tokenizer_demo.py
# 3. 初始化空模型
python init_zero_model.py
# 4. 预训练
bash pt.sh
