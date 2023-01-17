
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=qmix --env-config=equal_line with env_args.n_agents=5 env_args.map_name=equal_line_5 t_max=1000000 save_model=True
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=qmix --env-config=equal_line with env_args.n_agents=3 env_args.map_name=equal_line_3 collect_data=True checkpoint_path='/home/quy/mypymarl2/results/models/equal_line_3/qmix_env=8_adam_td_lambda__equal_line_3__parallel__2023-01-03_21-05-48' collect_nepisode=5000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=qmix --env-config=equal_line with env_args.n_agents=5 env_args.map_name=equal_line_5 collect_data=True checkpoint_path='/home/quy/mypymarl2/results/models/equal_line_5/qmix_env=8_adam_td_lambda__equal_line_5__parallel__2023-01-03_21-05-42' collect_nepisode=5000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=qmix --env-config=equal_line with env_args.n_agents=10 env_args.map_name=equal_line_10 collect_data=True checkpoint_path='/home/quy/mypymarl2/results/models/equal_line_10/qmix_env=8_adam_td_lambda__equal_line_10__parallel__2023-01-03_20-58-52' collect_nepisode=5000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cql_qmix --env-config=equal_line with env_args.n_agents=5 env_args.map_name=equal_line_5 raw_cql=True raw_sample_actions=500 training_episodes=1000 t_max=1000000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cql_qmix --env-config=equal_line with env_args.n_agents=5 env_args.map_name=equal_line_5 raw_cql=False raw_sample_actions=500 training_episodes=1000 t_max=1000000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cql_qmix --env-config=equal_line with env_args.n_agents=3 env_args.map_name=equal_line_3 raw_cql=True raw_sample_actions=500 training_episodes=1000 t_max=1000000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cql_qmix --env-config=equal_line with env_args.n_agents=3 env_args.map_name=equal_line_3 raw_cql=False raw_sample_actions=500 training_episodes=1000 t_max=1000000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cql_qmix --env-config=equal_line with env_args.n_agents=10 env_args.map_name=equal_line_10 raw_cql=True raw_sample_actions=500 training_episodes=1000 t_max=1000000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cql_qmix --env-config=equal_line with env_args.n_agents=10 env_args.map_name=equal_line_10 raw_cql=False raw_sample_actions=500 training_episodes=1000 t_max=1000000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=qmix --env-config=consensus with env_args.n_agents=5 env_args.map_name=consensus_5 t_max=1000000 save_model=True
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=qmix --env-config=consensus with env_args.n_agents=3 env_args.map_name=consensus_3 collect_data=True checkpoint_path='/home/quy/mypymarl2/results/models/equal_line_3/qmix_env=8_adam_td_lambda__equal_line_3__parallel__2022-12-29_16-50-35' collect_nepisode=5000
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cql_qmix --env-config=equal_line with env_args.n_agents=5 env_args.map_name=equal_line_5 raw_cql=True raw_sample_actions=500 training_episodes=1000


CUDA_VISIBLE_DEVICES=3 python3 src/main.py --config=cql_qmix --env-config=consensus with env_args.n_agents=5 env_args.map_name=consensus_5 raw_cql=True raw_sample_actions=250 batch_size=3 t_max=10000
CUDA_VISIBLE_DEVICES=4 python3 src/main.py --config=cql_qmix --env-config=consensus with env_args.n_agents=5 env_args.map_name=consensus_5 raw_sample_actions=250 batch_size=3 t_max=10000
CUDA_VISIBLE_DEVICES=5 python3 src/main.py --config=qmix --env-config=consensus with env_args.n_agents=5 env_args.map_name=consensus_5 batch_size=3 t_max=10000 use_offline=True
