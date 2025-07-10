export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:adroit_wsrl \
--project method-section \
--online_sampling_method append \
--num_offline_steps 300_000 \
--reward_scale 1.0 \
--reward_bias 0.0 \
--env NutAssemblySquare \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--data_path "/home/admin/ibrl/release/data/robomimic/square/processed_data96.hdf5" \
$@
