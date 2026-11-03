# If not using flash_attn, or transformers<4.44,
# or encountering an abnormally large loss (i.e., the model does not support packing),
# please remove `--packing true`.
nproc_per_node=2

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=6,7 \
MODELSCOPE_CACHE=/data/zhao/deploy/ms-swift/datasets \
swift pt \
    --model empty_qwen35_custom_init \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --tuner_type full \
    --dataset 'swift/chinese-c4#1000' \
    --torch_dtype bfloat16 \
    #--streaming true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    #--packing true \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 8192 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --output_dir output/pretrain \
    #--attn_impl flash_attn
