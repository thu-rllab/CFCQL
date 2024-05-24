from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .myenv import EqualLine,Consensus


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["equal_line"] = partial(env_fn, env=EqualLine)
REGISTRY["consensus"] = partial(env_fn, env=Consensus)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
