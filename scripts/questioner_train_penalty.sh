#!/bin/bash

solver_model_path=$1
questioner_model_path=$2
save_path=$3
echo "save_path: $save_path"
# 生成唯一 RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"
mkdir -p ${STORAGE_PATH}/sbgm/cognitive_graph ${STORAGE_PATH}/sbgm/semantic_space

# 启动 vllm 服务（记录 PID）
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# 开始训练 Questioner
echo "Start training questioner: $questioner_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller.py:compute_score \
    worker.reward.reward_function_kwargs.uncertainty_sigma=0.20 \
    worker.reward.reward_function_kwargs.repetition_tau=0.88 \
    worker.reward.reward_function_kwargs.repetition_pool_size=20000 \
    s_bgm.enabled=true \
    s_bgm.role=questioner \
    s_bgm.semantic_clusters_m=100 \
    s_bgm.uncertainty_intervals_n=10 \
    s_bgm.sliding_window_L=1024 \
    s_bgm.structural_entropy_K=2 \
    s_bgm.entropy_decay_gamma=0.01 \
    s_bgm.pi_kp=1.0 \
    s_bgm.pi_ki=0.01 \
    s_bgm.centroids_path=${STORAGE_PATH}/sbgm/semantic_space/centroids.npy \
    s_bgm.graph_state_path=${STORAGE_PATH}/sbgm/cognitive_graph/state.json \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.max_steps=6 \
    trainer.save_freq=1

sleep 5

# 合并模型
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

pkill python

echo "questioner training finished"
