# CFCQL-Continuous

This codebase is based on the open-source [OMAR](https://github.com/ling-pan/OMAR) framework, and please refer to that repo for more documentation.

## Requirements

- Multi-agent Particle Environments: in envs/multiagent-particle-envs and install it by `pip install -e .`
- python: 3.8
- torch > 1.4
- [baselines](https://github.com/openai/baselines)
- seaborn
- gym==0.10.8
- [MuJoCo==2.0](roboti.us/download.html): The dataset is sampled from MuJoCo2.0. Using mujoco>=2.1 will cause great performance drop
- mujoco_py200
- Multi-agent MuJoCo: Please check the [multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco) repo for more details about the environment. You can use the copy "multiagent_mujoco" in this directory without installation directly.

## Datasets
Datasets for different tasks are available at the following links uploaded by the author of [OMAR](https://github.com/ling-pan/OMAR). Please download the datasets and decompress them to the datasets folder.
- [HalfCheetah](https://drive.google.com/file/d/1zELoWUZoy3wPpwYni9t_TbzOjF4Px2f0/view?usp=sharing)
- [Cooperative Navigation](https://drive.google.com/file/d/1YVk_ajtvbcq8R2m0u0RasfB0csToV7XP/view?usp=sharing)
- [Predator-Prey](https://pan.baidu.com/s/16W-UyyCtfKDt9oTgeNOhJA): password is m7vw
- [World](https://pan.baidu.com/s/1pjZmeIAlaepPpug3b5olGA): password is 5k3t

Note: The datasets are too large, and the Baidu (Chinese) online disk requires a password for accessing it. Please just enter the password in the input box and click the blue button. The dataset can then be downloaded by cliking the "download" button (the second white button).

## Usage

Please follow the instructions below to replicate the results in the paper. 

```
python main.py --env_id <ENVIRONMENT_NAME> --data_type <DATA_TYPE> --dataset_num <DATASET_NUM> --cql --cf_cql --central_critic
```

- env_id: simple_spread/simple_tag/simple_world/HalfCheetah-v2
- data_type: random/medium-replay/medium/expert
- dataset_num: 0/1/2/3/4


