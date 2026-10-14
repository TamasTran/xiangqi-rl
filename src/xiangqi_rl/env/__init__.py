"""Gymnasium environment, observation/action encoders and rewards for Xiangqi."""

from xiangqi_rl.env.action_space import (
    ACTION_DIM,
    action_to_move,
    index_to_square,
    legal_action_mask,
    move_to_action,
    square_to_index,
)
from xiangqi_rl.env.environment import EpisodeOverError, IllegalActionError, XiangqiEnv
from xiangqi_rl.env.observation import (
    CHANNEL_NAMES,
    NUM_CHANNELS,
    OBSERVATION_SHAPE,
    encode_observation,
    observation_space,
    piece_channel,
)
from xiangqi_rl.env.rewards import result_for_side, step_reward

__all__ = [
    "ACTION_DIM",
    "CHANNEL_NAMES",
    "NUM_CHANNELS",
    "OBSERVATION_SHAPE",
    "EpisodeOverError",
    "IllegalActionError",
    "XiangqiEnv",
    "action_to_move",
    "encode_observation",
    "index_to_square",
    "legal_action_mask",
    "move_to_action",
    "observation_space",
    "piece_channel",
    "result_for_side",
    "square_to_index",
    "step_reward",
]
