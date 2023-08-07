
from .nq_learner import NQLearner
from .cq_learner import CQLearner
from .indq_learner import IndQLearner
from .icq_learner import ICQLearner
from .awac_learner import AWACLearner
from .iql_learner import IQLLearner
from .bc_learner import BCLearner
from .omar_learner import OMARLearner
from .baseline_madtkd_learner import MADTKDLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["cq_learner"] = CQLearner
REGISTRY["indq_learner"] = IndQLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["icq_learner"] = ICQLearner
REGISTRY["iql_learner"] = IQLLearner
REGISTRY["awac_learner"] = AWACLearner
REGISTRY["omar_learner"] = OMARLearner
REGISTRY["baseline_madtkd_learner"] = MADTKDLearner