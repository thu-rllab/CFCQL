REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .mpe_episode_runner import EpisodeRunner as MPEEpisodeRunner
REGISTRY["mpe_episode"] = MPEEpisodeRunner

from .mpe_parallel_runner import ParallelRunner as MPEParallelRunner
REGISTRY["mpe_parallel"] = MPEParallelRunner
