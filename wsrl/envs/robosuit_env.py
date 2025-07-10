import robosuite
from robosuite import load_controller_config
import numpy as np
import gym
import h5py



GOOD_CAMERAS = {
    "Lift": ["agentview", "sideview", "robot0_eye_in_hand"],
    "PickPlaceCan": ["agentview", "robot0_eye_in_hand"],
    "NutAssemblySquare": ["agentview", "robot0_eye_in_hand"],
}
DEFAULT_CAMERA = "agentview"


DEFAULT_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
STATE_KEYS = {
    "Lift": DEFAULT_STATE_KEYS,
    "PickPlaceCan": DEFAULT_STATE_KEYS,
    "NutAssemblySquare": DEFAULT_STATE_KEYS,
    "TwoArmTransport": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
        "object",
    ],
    "ToolHang": [
        "object",  # (389, 44)
        "robot0_eef_pos",  # (389, 3)
        "robot0_eef_quat",  # (389, 4)
        "robot0_gripper_qpos",  # (389, 2)
        # "robot0_gripper_qvel",  # (389, 2)
        # "robot0_eef_vel_ang",  # (389, 3)
        # "robot0_eef_vel_lin",  # (389, 3)
        # "robot0_joint_pos", # (389, 7)
        # "robot0_joint_pos_cos",  # (389, 7)
        # "robot0_joint_pos_sin",  # (389, 7)
        # "robot0_joint_vel",  # (389, 7)
    ],
}
STATE_SHAPE = {
    "Lift": (19,),
    "PickPlaceCan": (23,),
    "NutAssemblySquare": (23,),
    "TwoArmTransport": (59,),
    "ToolHang": (53,),
}

class robosuite_env:
    def __init__(self,env_name:str,reward_scale,reward_bias) -> None:
        controller_config = load_controller_config(default_controller="OSC_POSE")
        self.obs_keys = STATE_KEYS[env_name]
        self.obs_keys = ["object-state" if s == "object" else s for s in self.obs_keys]
        self.env = robosuite.make(
            env_name=env_name,
            robots=["Panda"],             # load a Sawyer robot and a Panda robot
            gripper_types="default",                # use default grippers per robot arm
            controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
            has_renderer=False,                     # no on-screen rendering
            has_offscreen_renderer=False,           # no off-screen rendering
            control_freq=20,                        # 20 hz control for applied actions
            horizon=300,                            # each episode terminates after 200 steps
            use_object_obs=True,                    # provide object observations to agent
            use_camera_obs=False,                   # don't provide image observations to agent
            reward_shaping=False,                    # use a dense reward signal for learning
        )
        #TODO observation space and action space
        self.observation_space = gym.spaces.Box(low=-1.0,high=1.0,shape = (3*STATE_SHAPE[env_name][0],),dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(7,),dtype=np.float32)
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.pobs = np.zeros(STATE_SHAPE[env_name])
        self.ppobs = np.zeros(STATE_SHAPE[env_name])
        self.single_obs_shape = STATE_SHAPE[env_name]

    def reset(self):
        #empty the pobs and ppobs
        self.pobs = np.zeros(self.single_obs_shape)
        self.ppobs = np.zeros(self.single_obs_shape)
        obs_dict = self.env.reset()
        obs = self._flat_obs(obs_dict)
        info = {"goal_achieved":0}
        obs = self._stack_obs(obs)
        return obs,info
    
    def step(self,action:np.ndarray):
        obs_dict,reward,done,info = self.env.step(action)
        if reward == 1:
            info = {"goal_achieved":1}
        else:
            info = {"goal_achieved":0}
        reward = reward * self.reward_scale + self.reward_bias
        obs = self._flat_obs(obs_dict)
        obs = self._stack_obs(obs)
        return obs,reward,done,done,info
    
    def _flat_obs(self,obs_dict):
        return np.concatenate([obs_dict[k] for k in self.obs_keys])

    def _stack_obs(self,obs):
        obs_stacked = stack_three(obs,self.pobs,self.ppobs)
        self.ppobs = self.pobs
        self.pobs = obs
        return obs_stacked


def get_robosuite_dataset(data_path:str,env_name:str,reward_scale,reward_bias,num_data=50):
    f = h5py.File(data_path)
    num_episode: int = len(list(f["data"].keys()))  # type: ignore
    print(f"loading first {num_data} episodes from {data_path}")
    print(f"Raw Dataset size (#episode): {num_episode}")

    all_actions = []
    all_states = []
    all_rewards = []
    all_dones = []
    all_next_states = []
    for episode_id in range(num_episode):
        if num_data > 0 and episode_id >= num_data:
            break

        episode_tag = f"demo_{episode_id}"
        episode = f[f"data/{episode_tag}"]
        assert True,f'{episode.keys()}'

        actions = np.array(episode["actions"]).astype(np.float32)  # type: ignore

        temp_states = []
        for key in STATE_KEYS[env_name]:
            state_: np.ndarray = episode["obs"][key]  # type: ignore
            temp_states.append(state_)
        states = np.concatenate(temp_states, axis=1)
        assert states.shape[0] == actions.shape[0]

        #stack states
        pstates = np.zeros_like(states)
        ppstates = np.zeros_like(states)
        pstates[1:] = states[0:-1]
        ppstates[2:] = states[0:-2]
        states = stack_three(states,pstates,ppstates)

        rewards = np.array(episode["rewards"])
        rewards = rewards * reward_scale + reward_bias
        dones = np.array(episode["dones"])
        
        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]
        done_index = np.argmax(dones == 1)

        all_states.append(states[:done_index+1])
        all_actions.append(actions[:done_index+1])
        all_rewards.append(rewards[:done_index+1])
        all_dones.append(dones[:done_index+1])
        all_next_states.append(next_states[:done_index+1])
        assert True,f'{actions[-20:],dones[:done_index+1],rewards[:done_index+1]}'
    
    states = np.concatenate(all_states,axis=0)
    actions = np.concatenate(all_actions,axis=0)
    rewards = np.concatenate(all_rewards,axis = 0)
    dones = np.concatenate(all_dones,axis=0)
    next_states = np.concatenate(all_next_states,axis=0)
    assert True,f'{states[-2:],next_states[-3:],dones[-3:],rewards[-3:]}'
    return dict(
        observations=states,
        actions=actions,
        next_observations=next_states,
        rewards=rewards,
        dones=dones,
        masks=1 - dones,
    )

def get_robosuite_dataset_mc(data_path:str,env_name:str,reward_scale,reward_bias,num_data=50):
    f = h5py.File(data_path)
    num_episode: int = len(list(f["data"].keys()))  # type: ignore
    print(f"loading first {num_data} episodes from {data_path}")
    print(f"Raw Dataset size (#episode): {num_episode}")

    all_actions = []
    all_states = []
    all_rewards = []
    all_dones = []
    all_next_states = []
    all_mc = []
    for episode_id in range(num_episode):
        if num_data > 0 and episode_id >= num_data:
            break

        episode_tag = f"demo_{episode_id}"
        episode = f[f"data/{episode_tag}"]
        assert True,f'{episode.keys()}'

        actions = np.array(episode["actions"]).astype(np.float32)  # type: ignore

        temp_states = []
        for key in STATE_KEYS[env_name]:
            state_: np.ndarray = episode["obs"][key]  # type: ignore
            temp_states.append(state_)
        states = np.concatenate(temp_states, axis=1)
        assert states.shape[0] == actions.shape[0]

        rewards = np.array(episode["rewards"])
        rewards = rewards * reward_scale + reward_bias
        dones = np.array(episode["dones"])
        
        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]
        #calculate mc return
        mc = np.zeros_like(states)
        prev_return = 0
        for i in reversed(range(len(states))):
            mc[i] = rewards[i] + 0.99 * (1-dones[i])*prev_return
            prev_return = mc[i]
        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_dones.append(dones)
        all_next_states.append(next_states)
        all_mc.append(mc)
        assert True,f'{dones,rewards}'
    
    states = np.concatenate(all_states,axis=0)
    actions = np.concatenate(all_actions,axis=0)
    rewards = np.concatenate(all_rewards,axis = 0)
    dones = np.concatenate(all_dones,axis=0)
    next_states = np.concatenate(all_next_states,axis=0)
    mcs = np.concatenate(all_mc,axis=0)
    assert True,f'{states[-2:],next_states[-3:],dones[-3:],rewards[-3:]}'
    return dict(
        observations=states,
        actions=actions,
        next_observations=next_states,
        rewards=rewards,
        dones=dones,
        masks=1 - dones,
        mc_returns = mcs,
    )

def stack_three(obs,pobs,ppobs):
    return np.concatenate([obs,pobs,ppobs],axis=-1)

