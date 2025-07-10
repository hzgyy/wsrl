import os
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from experiments.configs.ensemble_config import add_redq_config
from wsrl.agents import agents
from wsrl.common.evaluation import evaluate_with_trajectories
from wsrl.common.wandb import WandBLogger
from wsrl.data.replay_buffer import ReplayBuffer, ReplayBufferMC
from wsrl.envs.adroit_binary_dataset import get_hand_dataset_with_mc_calculation
from wsrl.envs.d4rl_dataset import (
    get_d4rl_dataset,
    get_d4rl_dataset_with_mc_calculation,
)
from wsrl.envs.env_common import get_env_type, make_gym_env
from wsrl.utils.timer_utils import Timer
from wsrl.utils.train_utils import concatenate_batches, subsample_batch
from wsrl.envs.robosuit_env import robosuite_env, get_robosuite_dataset, get_robosuite_dataset_mc
from jax import default_backend

FLAGS = flags.FLAGS

# env
flags.DEFINE_string("env", "antmaze-large-diverse-v2", "Environemnt to use")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -1.0, "Reward bias.")
flags.DEFINE_float(
    "clip_action",
    0.99999,
    "Clip actions to be between [-n, n]. This is needed for tanh policies.",
)

# training
flags.DEFINE_integer("num_offline_steps", 1_000_000, "Number of offline epochs.")
flags.DEFINE_integer("num_online_steps", 500_000, "Number of online epochs.")
flags.DEFINE_float(
    "offline_data_ratio",
    0.0,
    "How much offline data to retain in each online batch update",
)
flags.DEFINE_string(
    "online_sampling_method",
    "mixed",
    """Method of sampling data during online update: mixed or append.
    `mixed` samples from a mix of offline and online data according to offline_data_ratio.
    `append` adds offline data to replay buffer and samples from it.""",
)
flags.DEFINE_bool(
    "online_use_cql_loss",
    True,
    """When agent is CQL/CalQL, whether to use CQL loss for the online phase (use SAC loss if False)""",
)
flags.DEFINE_integer(
    "warmup_steps", 0, "number of warmup steps (WSRL) before performing online updates"
)
flags.DEFINE_string("data_path", None, "data path to the offline training data")

# agent
flags.DEFINE_string("agent", "calql", "what RL agent to use")
flags.DEFINE_integer("utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_integer("batch_size", 256, "batch size for training")
flags.DEFINE_integer("replay_buffer_capacity", int(2e6), "Replay buffer capacity")
flags.DEFINE_bool("use_redq", False, "Use an ensemble of Q-functions for the agent")

# experiment house keeping
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "save_dir",
    os.path.expanduser("~/wsrl_log"),
    "Directory to save the logs and checkpoints",
)
flags.DEFINE_string("resume_path", "", "Path to resume from")
flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 100_000, "Save every n steps.")
flags.DEFINE_integer(
    "n_eval_trajs", 20, "Number of trajectories to use for each evaluation."
)
flags.DEFINE_bool("deterministic_eval", True, "Whether to use deterministic evaluation")

# wandb
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", None, "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    """
    house keeping
    """
    assert FLAGS.online_sampling_method in [
        "mixed",
        "append",
    ], "incorrect online sampling method"

    if FLAGS.use_redq:
        FLAGS.config.agent_kwargs = add_redq_config(FLAGS.config.agent_kwargs)

    min_steps_to_update = FLAGS.batch_size * (1 - FLAGS.offline_data_ratio)
    if FLAGS.agent == "calql":
        # min_steps_to_update = max(
        #     min_steps_to_update, gym.make(FLAGS.env)._max_episode_steps
        # )
        min_steps_to_update = max(
            min_steps_to_update, 300
        )
    """
    wandb and logging
    """
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "wsrl" or FLAGS.project,
            "group": "wsrl" or FLAGS.group,
            "exp_descriptor": f"{FLAGS.exp_name}_{FLAGS.env}_{FLAGS.agent}_seed{FLAGS.seed}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        random_str_in_identifier=True,
        disable_online_logging=FLAGS.debug,
    )

    save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    """
    env
    """
    # do not clip adroit actions online following CalQL repo
    # https://github.com/nakamotoo/Cal-QL
    env_type = get_env_type(FLAGS.env)
    if FLAGS.env in ("NutAssemblySquare","Lift","PickPlaceCan","TwoArmTransport","ToolHang"):
        finetune_env = robosuite_env(FLAGS.env,FLAGS.reward_scale,FLAGS.reward_bias)
        eval_env = robosuite_env(FLAGS.env,FLAGS.reward_scale,FLAGS.reward_bias)
        # pass
    else:
        finetune_env = make_gym_env(
            env_name=FLAGS.env,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
            action_clip_lim=FLAGS.clip_action,
            seed=FLAGS.seed,
        )
        eval_env = make_gym_env(
            env_name=FLAGS.env,
            scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
            action_clip_lim=FLAGS.clip_action,
            seed=FLAGS.seed + 1000,
        )
    # obs,_ = finetune_env.reset()
    # action = np.random.uniform(low=-1.0, high=1.0, size=(7,))
    # nobs,reward,done,truncated,info = finetune_env.step(action)
    
    assert True,f'{nobs.shape}'
    
    """
    load dataset
    """
    if env_type == "adroit-binary":
        dataset = get_hand_dataset_with_mc_calculation(
            FLAGS.env,
            gamma=FLAGS.config.agent_kwargs.discount,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
        )
    elif env_type == "robosuite":
        if FLAGS.agent == "calql":
            dataset = get_robosuite_dataset_mc(FLAGS.data_path,FLAGS.env,FLAGS.reward_scale,FLAGS.reward_bias,50)
        else:
            dataset = get_robosuite_dataset(FLAGS.data_path,FLAGS.env,FLAGS.reward_scale,FLAGS.reward_bias,50)
        # pass
    else:
        if FLAGS.agent == "calql":
            # need dataset with mc return
            dataset = get_d4rl_dataset_with_mc_calculation(
                FLAGS.env,
                reward_scale=FLAGS.reward_scale,
                reward_bias=FLAGS.reward_bias,
                clip_action=FLAGS.clip_action,
                gamma=FLAGS.config.agent_kwargs.discount,
            )
        else:
            dataset = get_d4rl_dataset(
                FLAGS.env,
                reward_scale=FLAGS.reward_scale,
                reward_bias=FLAGS.reward_bias,
                clip_action=FLAGS.clip_action,
            )

    """
    replay buffer
    """
    replay_buffer_type = ReplayBufferMC if FLAGS.agent == "calql" else ReplayBuffer
    replay_buffer = replay_buffer_type(
        finetune_env.observation_space,
        finetune_env.action_space,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
        discount=FLAGS.config.agent_kwargs.discount if FLAGS.agent == "calql" else None,
    )

    """
    Initialize agent
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    example_batch = subsample_batch(dataset, FLAGS.batch_size)
    agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
    )

    ref_agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
    )
    if FLAGS.resume_path != "":
        assert os.path.exists(FLAGS.resume_path), "resume path does not exist"
        agent = checkpoints.restore_checkpoint(FLAGS.resume_path, target=agent)
        ref_agent = checkpoints.restore_checkpoint(FLAGS.resume_path, target=ref_agent)

    """
    eval function
    """

    def evaluate_and_log_results(
        eval_env,
        policy_fn,
        eval_func,
        step_number,
        wandb_logger,
        n_eval_trajs=FLAGS.n_eval_trajs,
    ):
        stats, trajs = eval_func(
            policy_fn,
            eval_env,
            n_eval_trajs,
        )

        eval_info = {
            "average_return": np.mean([np.sum(t["rewards"]) for t in trajs]),
            "average_traj_length": np.mean([len(t["rewards"]) for t in trajs]),
        }
        if env_type == "adroit-binary" or env_type == "robosuite":
            # adroit
            eval_info["success_rate"] = np.mean(
                [any(d["goal_achieved"] for d in t["infos"]) for t in trajs]
            )
        elif env_type == "kitchen":
            # kitchen
            eval_info["num_stages_solved"] = np.mean([t["rewards"][-1] for t in trajs])
            eval_info["success_rate"] = np.mean([t["rewards"][-1] for t in trajs]) / 4
        else:
            # d4rl antmaze, locomotion
            eval_info["success_rate"] = eval_info[
                "average_normalized_return"
            ] = np.mean(
                [eval_env.get_normalized_score(np.sum(t["rewards"])) for t in trajs]
            )
        
        return eval_info
        # wandb_logger.log({"evaluation": eval_info}, step=step_number)
    """
    training loop
    """
    timer = Timer()
    step = int(agent.state.step)  # 0 for new agents, or load from pre-trained
    is_online_stage = False
    observation, info = finetune_env.reset()
    done = False  # env done signal
    assert True,f"total:{FLAGS.num_offline_steps + FLAGS.num_online_steps}----------------------"
    for _ in tqdm.tqdm(range(step, FLAGS.num_offline_steps + FLAGS.num_online_steps)):
        """
        Switch from offline to online
        """
        # print("JAX backend:", default_backend())
        if not is_online_stage and step >= FLAGS.num_offline_steps:
            logging.info("Switching to online training")
            is_online_stage = True

            # upload offline data to online buffer
            if FLAGS.online_sampling_method == "append":
                offline_dataset_size = dataset["actions"].shape[0]
                dataset_items = dataset.items()
                for j in range(offline_dataset_size):
                    if FLAGS.agent == 'calql':
                        transition = {k: v[j] for k, v in dataset_items}
                    else:
                        transition = {k: v[j] for k, v in dataset_items if k != 'mc_returns'}
                    replay_buffer.insert(transition)

            # option for CQL and CalQL to change the online alpha, and whether to use CQL regularizer
            if FLAGS.agent in ("cql", "calql"):
                online_agent_configs = {
                    "cql_alpha": FLAGS.config.agent_kwargs.get(
                        "online_cql_alpha", None
                    ),
                    "use_cql_loss": FLAGS.online_use_cql_loss,
                }
                agent.update_config(online_agent_configs)

        timer.tick("total")

        """
        Env Step
        """
        with timer.context("env step"):
            if is_online_stage:
                rng, action_rng = jax.random.split(rng)
                action = agent.sample_actions(observation, seed=action_rng)
                next_observation, reward, done, truncated, info = finetune_env.step(
                    action
                )
                transition = dict(
                    observations=observation,
                    next_observations=next_observation,
                    actions=action,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=1.0 if (done or truncated) else 0,
                )
                replay_buffer.insert(transition)

                observation = next_observation
                if done or truncated:
                    observation, info = finetune_env.reset()
                    done = False

        """
        Updates
        """
        with timer.context("update"):
            # offline updates
            if not is_online_stage:
                batch = subsample_batch(dataset, FLAGS.batch_size)
                agent, update_info = agent.update(
                    batch,
                )

            # online updates
            else:
                if step - FLAGS.num_offline_steps <= max(
                    FLAGS.warmup_steps, min_steps_to_update
                ):
                    # no updates during warmup
                    pass
                else:
                    # do online updates, gather batch
                    if FLAGS.online_sampling_method == "mixed":
                        # batch from a mixing ratio of offline and online data
                        batch_size_offline = int(
                            FLAGS.batch_size * FLAGS.offline_data_ratio
                        )
                        batch_size_online = FLAGS.batch_size - batch_size_offline
                        online_batch = replay_buffer.sample(batch_size_online)
                        offline_batch = subsample_batch(dataset, batch_size_offline)
                        # update with the combined batch
                        batch = concatenate_batches([online_batch, offline_batch])
                    elif FLAGS.online_sampling_method == "append":
                        # batch from online replay buffer, with is initialized with offline data
                        batch = replay_buffer.sample(FLAGS.batch_size)
                    else:
                        raise RuntimeError("Incorrect online sampling method")

                    # update
                    if FLAGS.utd > 1:
                        agent, update_info = agent.update_high_utd(
                            batch,
                            utd_ratio=FLAGS.utd,
                        )
                    else:
                        agent, update_info = agent.update(
                            batch,
                        )

        """
        Advance Step
        """
        step += 1

        """
        Evals
        """
        eval_steps = (
            FLAGS.num_offline_steps,  # finish offline training
            FLAGS.num_offline_steps + 1,  # start of online training
            FLAGS.num_offline_steps + FLAGS.num_online_steps,  # end of online training
        )
        if step % FLAGS.eval_interval == 0 or step in eval_steps:
            logging.info("Evaluating...")
            with timer.context("evaluation"):
                policy_fn = partial(
                    agent.sample_actions, argmax=FLAGS.deterministic_eval
                )
                eval_func = partial(
                    evaluate_with_trajectories, clip_action=FLAGS.clip_action
                )

                eval_info = evaluate_and_log_results(
                    eval_env=eval_env,
                    policy_fn=policy_fn,
                    eval_func=eval_func,
                    step_number=step,
                    wandb_logger=wandb_logger,
                )
                
                #calculate q
                # off_batch = subsample_batch(dataset, 512)
                # on_batch = replay_buffer.sample(512)
                # observations = off_batch["observations"]
                # off_qs = agent.forward_value(observations,train=False)
                # eval_info["off_q"] = off_qs.mean()
                # off_qs_ref = ref_agent.forward_value(observations,train=False)
                # eval_info["delta_q"] = (off_qs-off_qs_ref).mean()
                # #calculate variance
                # on_q,on_qs = agent.forward_values(on_batch["observations"],train=False)
                # eval_info["on_q"] = on_q.mean()
                # print(f'on_qs:{on_qs.shape}')
                # on_stds = jnp.std(on_qs,axis=0,ddof=1)
                # print(f'on_std:{on_stds.shape}')
                # eval_info["on_q_std_mean"] = on_stds.mean()
                # eval_info["on_q_std_max"] = on_stds.max()
                # eval_info["on_q_std_min"] = on_stds.min()
                # # calculate q kl
                # logp = jax.nn.log_softmax(off_qs_ref, axis=0)  # log π_ref
                # logq = jax.nn.log_softmax(off_qs, axis=0)      # log π
                # p = jnp.exp(logp)                                    # π_ref
                # q_kl = jnp.sum(p * (logp - logq), axis=0)  # shape [B]
                # eval_info["q_kl"] = jnp.mean(q_kl)
                # # calculate act kl
                # rng, rng_dist, rng_ref = jax.random.split(rng, 3)
                # dist = agent.forward_policy(observations, rng=rng_dist, train=False)
                # dist_ref = ref_agent.forward_policy(observations, rng=rng_ref, train=False)
                # kl = kl_divergence_normal_diag(dist_ref, dist)
                # # act_ref,logp_ref = ref_agent.forward_policy_and_sample(observations,eval_action_rng,repeat=10) #(batch,repeat,action_dim),(batch,repeat)
                # # logp = dist.log_prob()
                # # kl_per_sample = jnp.mean(logp_ref - logp, axis=1)
                # eval_info["act_kl"] = jnp.mean(kl)
                logging.info(f"current step:{step}!!!!!!!!!!!!!!!!!!!!!!!!")
                wandb_logger.log({"evaluation": eval_info}, step=step)


        """
        Save Checkpoint
        """
        if step % FLAGS.save_interval == 0 or step == FLAGS.num_offline_steps:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=step, keep=30
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        """
        Logging
        """
        if step % FLAGS.log_interval == 0:
            # check if update_info is available (False during warmup)
            if "update_info" in locals():
                update_info = jax.device_get(update_info)
                wandb_logger.log({"training": update_info}, step=step)

            wandb_logger.log({"timer": timer.get_average_times()}, step=step)


def kl_divergence_normal_diag(dist_ref, dist):
    base_ref = dist_ref.distribution
    base = dist.distribution
    mu1, std1 = base_ref.loc, base_ref.scale_diag
    mu2, std2 = base.loc, base.scale_diag

    var1 = std1 ** 2
    var2 = std2 ** 2

    kl_per_dim = (
        (var1 / var2)
        + ((mu2 - mu1) ** 2) / var2
        - 1
        + 2 * (jnp.log(std2) - jnp.log(std1))
    )

    kl = 0.5 * jnp.sum(kl_per_dim, axis=-1)  # shape: [B]
    return kl

if __name__ == "__main__":
    app.run(main)
