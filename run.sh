CUDA_VISIBLE_DEVICES=0 python generate_topic_intent_test.py \
    --model_name vicuna-7b \
    --model_path /home/luxu/yan/pytorch_models/vicuna-7b-v1.5 \
    --batch_size 5 \
    --prompt only_topic \
    --loc '1' \
    --begin_index 50 \
    --split_size 150 \
    --data_path test.jsonl \
    --save_path ./results_v1/
