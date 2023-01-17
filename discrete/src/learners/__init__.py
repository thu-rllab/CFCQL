
from .nq_learner import NQLearner
from .cq_learner import CQLearner
from .icq_learner import ICQLearner
from .bc_learner import BCLearner
from .omar_learner import OMARLearner
from .baseline_madtkd_learner import MADTKDLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["cq_learner"] = CQLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["icq_learner"] = ICQLearner
REGISTRY["omar_learner"] = OMARLearner
REGISTRY["baseline_madtkd_learner"] = MADTKDLearner