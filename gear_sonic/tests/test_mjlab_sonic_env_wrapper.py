import torch

from gear_sonic.envs.wrapper.mjlab_sonic_env_wrapper import MjlabSonicEnvWrapper


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _DictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


class _FakeMjlabEnv:
    device = "cpu"
    num_envs = 2

    def __init__(self):
        self.observation_space = _DictSpace(
            {
                "actor": _Space((self.num_envs, 5)),
                "critic": _Space((self.num_envs, 7)),
            }
        )
        self.action_space = _Space((self.num_envs, 29))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.command_manager = object()

    def reset(self):
        return (
            {
                "actor": torch.zeros(self.num_envs, 5),
                "critic": torch.zeros(self.num_envs, 7),
            },
            {"log": {}},
        )

    def step(self, action):
        assert action.shape == (self.num_envs, 29)
        return (
            {
                "actor": torch.ones(self.num_envs, 5),
                "critic": torch.ones(self.num_envs, 7),
            },
            torch.ones(self.num_envs),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
            {"log": {"reward": torch.ones(self.num_envs)}},
        )

    def close(self):
        return None


def test_mjlab_wrapper_translates_obs_and_step_contract():
    wrapper = MjlabSonicEnvWrapper(_FakeMjlabEnv())

    obs = wrapper.reset_all()
    assert set(obs) == {"actor_obs", "critic_obs"}
    assert obs["actor_obs"].shape == (2, 5)
    assert obs["critic_obs"].shape == (2, 7)
    assert wrapper.config.robot.algo_obs_dim_dict.actor_obs == 5
    assert wrapper.config.robot.algo_obs_dim_dict.critic_obs == 7
    assert wrapper.config.robot.actions_dim == 29
    assert wrapper.observation_space["policy"].shape == (2, 5)

    obs, rewards, dones, infos = wrapper.step({"actions": torch.zeros(2, 29)})
    assert obs["actor_obs"].sum().item() == 10
    assert rewards.tolist() == [1.0, 1.0]
    assert dones.tolist() == [1, 1]
    assert "episode" in infos
    assert infos["time_outs"].tolist() == [True, False]

