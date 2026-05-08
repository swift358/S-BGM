solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo $STORAGE_PATH
mkdir -p ${STORAGE_PATH}/sbgm/cognitive_graph ${STORAGE_PATH}/sbgm/semantic_space

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $questioner_model_path 1000 $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
local_train_file=${STORAGE_PATH}/datasets/${experiment_name}/train.parquet
echo "prepare local training dataset: ${local_train_file}"
python question_evaluate/upload.py \
    --repo_name "" \
    --uncertainty_delta 0.20 \
    --num_uncertainty_intervals 10 \
    --experiment_name ${experiment_name} \
    --local_train_file ${local_train_file}
echo 'start train'

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${local_train_file} \
    trainer.total_epochs=100 \
    trainer.max_steps=15 \
    s_bgm.enabled=true \
    s_bgm.role=solver \
    s_bgm.semantic_clusters_m=100 \
    s_bgm.uncertainty_intervals_n=10 \
    s_bgm.sliding_window_L=1024 \
    s_bgm.structural_entropy_K=2 \
    s_bgm.entropy_decay_gamma=0.01 \
    s_bgm.pi_kp=1.0 \
    s_bgm.pi_ki=0.01 \
    s_bgm.centroids_path=${STORAGE_PATH}/sbgm/semantic_space/centroids.npy \
    s_bgm.graph_state_path=${STORAGE_PATH}/sbgm/cognitive_graph/state.json \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=4 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor

sleep 10

echo "solver training finished"

# bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
# 不要崩溃啊！！！
