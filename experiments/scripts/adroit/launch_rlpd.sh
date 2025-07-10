export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:adroit_wsrl \
--project baselines-section \
--online_sampling_method mixed \
--num_offline_steps 20_000 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--offline_data_ratio 0.5 \
--utd 4 \
--batch_size $((256 * 4)) \
--warmup_steps 0 \
$@
