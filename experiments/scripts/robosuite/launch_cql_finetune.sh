export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0

python finetune.py \
--agent cql \
--config experiments/configs/train_config.py:adroit_cql \
--project baselines-section \
--group no-redq-utd1 \
--warmup_steps 0 \
--num_offline_steps 300_000 \
--save_interval 20_000 \
--eval_interval 20_000 \
--reward_scale 1.0 \
--reward_bias 0.0 \
--env "NutAssemblySquare" \
--data_path "/home/admin/ibrl/release/data/robomimic/square/processed_data96.hdf5" \
$@