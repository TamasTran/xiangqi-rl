"""Play agents against each other through :class:`XiangqiEnv`.

The runner owns the turn sequence: it asks the agent of ``env.side_to_move``
for exactly one action, steps the environment once, and repeats, so no agent
can ever move twice in a row. Colours are taken from the environment, never
assumed.
"""

from __future__ import annotations

from xiangqi_rl.agents.base import Agent
from xiangqi_rl.engine import GameConfig, GameResult, Side
from xiangqi_rl.env import IllegalActionError, XiangqiEnv
from xiangqi_rl.evaluation.metrics import GameRecord, MatchStats, summarize


def play_game(
    red: Agent,
    black: Agent,
    env: XiangqiEnv,
    seed: int | None = None,
) -> GameRecord:
    """Play one game with ``red`` and ``black`` and return its record.

    An illegal action forfeits the game for the offending agent (counted in
    ``invalid_actions``) instead of crashing the whole tournament.
    """
    agents = {Side.RED: red, Side.BLACK: black}
    env.reset(seed=seed)
    terminated = truncated = False
    invalid = 0
    forfeit_by: Side | None = None
    while not (terminated or truncated):
        side = env.side_to_move
        action = agents[side].select_action(env.game_copy())
        try:
            _, _, terminated, truncated, _ = env.step(action)
        except IllegalActionError:
            invalid += 1
            forfeit_by = side
            break

    state = env.state
    if forfeit_by is not None:
        result = GameResult.win_for(forfeit_by.opponent)
        termination = "illegal_action"
    else:
        result = state.result
        termination = state.termination.value if state.termination else None
    return GameRecord(
        red=red.name,
        black=black.name,
        result=result,
        plies=state.ply,
        truncated=truncated,
        invalid_actions=invalid,
        forfeit_by=forfeit_by,
        termination="external_step_limit" if truncated else termination,
    )


def run_match(
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
    *,
    seed: int = 0,
    swap_colors: bool = True,
    game_config: GameConfig | None = None,
    max_episode_steps: int | None = None,
) -> tuple[MatchStats, list[GameRecord]]:
    """Play ``num_games`` between ``agent_a`` and ``agent_b``.

    With ``swap_colors`` the agents alternate colours (``agent_a`` is RED in
    even-numbered games); otherwise ``agent_a`` is always RED. Game ``i`` is
    reset with ``seed + i``. Agent names must differ so statistics can be
    attributed; give two copies of one agent type distinct ``name``s.
    """
    if num_games < 0:
        raise ValueError(f"num_games must be non-negative, got {num_games}")
    if agent_a.name == agent_b.name:
        raise ValueError(f"Agents must have distinct names, both are {agent_a.name!r}")
    env = XiangqiEnv(game_config=game_config, max_episode_steps=max_episode_steps)
    records: list[GameRecord] = []
    for i in range(num_games):
        a_is_red = (i % 2 == 0) or not swap_colors
        red, black = (agent_a, agent_b) if a_is_red else (agent_b, agent_a)
        records.append(play_game(red, black, env, seed=seed + i))
    env.close()
    return summarize(records), records
