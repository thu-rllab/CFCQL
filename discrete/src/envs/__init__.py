from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .myenv import EqualLine,Consensus,MMDP,Spread
from .hallway import HallWayEnv
from .lbforaging import ForagingEnv

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except:
    gfootball = False

try:
    ic3netenv = True
    from .ic3netenvs import GymWrapper
except:
    ic3netenv = False
try:
    mpeenv = True
    from .mpe import MPEEnv
    from .pan_mpe import MPEEnv as PanMPEEnv
    # from .multiagent-particle-envs.make_env import make_env
except:
    mpeenv = False

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["equal_line"] = partial(env_fn, env=EqualLine)
REGISTRY["consensus"] = partial(env_fn, env=Consensus)
REGISTRY["mmdp"] = partial(env_fn, env=MMDP)
REGISTRY["spread"] = partial(env_fn, env=Spread)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

REGISTRY["foraging"] = partial(env_fn, env=ForagingEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallWayEnv)


if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)
if ic3netenv:
    REGISTRY["traffic_junction"] = partial(env_fn, env=GymWrapper)
    REGISTRY["predator_prey"] = partial(env_fn, env=GymWrapper)

if mpeenv:
    REGISTRY["mpe_env"] = partial(env_fn, MPEEnv)
    REGISTRY["pan_mpe_env"] = partial(env_fn, PanMPEEnv)
    # REGISTRY["mpe_env"] = partial(env_fn, make_env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
