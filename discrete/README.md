This repository is based on the [pymarl2](https://github.com/hijkzzz/pymarl2).

## Installation instructions

```shell
cd install
```

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
bash install_dependencies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

## Datasets

Datasets for sc2 are available at the following links. Please download the datasets and decompress them to the "offline_datasets" folder.

- [sc2datasets](https://drive.google.com/file/d/1Hn0CxnGDiwF9i7ugiLCyUYz-GzIkdQeD/view?usp=sharing)


## Command Line Tool

**Collect Datasets**

```shell
# Pretrain qmix model in 2s3z
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z save_model=True
```

```shell
# Collect Medium-Replay dataset in 2s3z
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z h5file_suffix=medium_replay use_offline=False t_max=500000 
```

```shell
# Collect Medium (or Expert) dataset in 2s3z
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z collect_data=True h5file_suffix='medium' collect_nepisode=5000 checkpoint_path=$checkpoint_path$ load_step=100000
```

```shell
# Collect Mixed dataset in 2s3z
cd tools
python mixed.py
```

**Train Offline**

```shell
#For CFCQL
python3 src/main.py --config=cql_qmix --env-config=sc2 with env_args.map_name=2s3z  h5file_suffix=medium global_cql_alpha=0.1  moderate_lambda=True softmax_temp=0.5
```

```shell
#For MACQL
python3 src/main.py --config=cql_qmix --env-config=sc2 with env_args.map_name=2s3z  h5file_suffix=medium global_cql_alpha=5.0  raw_cql=True
```

**Hyperparameters**

| **Map**         | **Dataset**      | **global\_cql\_alpha** | **softmax\_temp**    |
|:-----------------:|:------------------:|:-----------------------:|:----------------------:|
| **2s3z**        | medium           | 0.1                   | 0.5                  |
|                 | medium\_replay   | 5                     | 0.1                  |
|                 | expert           | 50                    | 0.5                  |
|                 | mixed            | 0.5                   | 0                    |
| **3s\_vs\_5z**  | medium           | 50                    | 0                    |
|                 | medium\_replay   | 50                    | 0                    |
|                 | expert           | 50                    | 1                    |
|                 | mixed            | 50                    | 0                    |
| **5m\_vs\_6m**  | medium           | 50                    | 0.5                  |
|                 | medium\_replay   | 50                    | 0.5                  |
|                 | expert           | 50                    | 100 (argmax)         |
|                 | mixed            | 50                    | 0.5                  |
| **6h\_vs\_8z**  | medium           | 50                    | 1                    |
|                 | medium\_replay   | 50                    | 0.5                  |
|                 | expert           | 50                    | 0                    |
|                 | mixed            | 50                    | 1                    |

