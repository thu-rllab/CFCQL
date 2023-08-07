import datetime
import os
import pprint
from re import L
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import numpy as np
from smac.env import StarCraft2Env
from functools import partial
from components.episode_buffer import EpisodeBatch
import h5py
from run.offline_utils import *
def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    if args.use_offline:
        tb_suffix = args.h5file_suffix
        if 'madtkd' in args.name:
            args.name=args.name+'_'+args.model_type
            if args.teacher:
                args.name='teacher_'+args.name+'_'
        if getattr(args, 'raw_cql', False):
            args.name = 'raw_'+args.name
            args.sparse_lambda = False
        if getattr(args, 'sparse_lambda', False):
            args.name = 'slsoftmaxkl_'+str(args.softmax_temp)+'_'+str(args.training_episodes)+'_'+args.name
        if  getattr(args, 'global_cql_alpha', False):
            tb_suffix += '_global_cql_alpha_'+str(args.global_cql_alpha)+'_'
        elif getattr(args, 'cql_alpha', False):
            tb_suffix += '_cql_alpha_'+str(args.cql_alpha)+'_'
        

        unique_token = "{}__{}__{}__{}__{}".format(args.name,args.env_args["map_name"],tb_suffix,args.runner, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        unique_token = "{}__{}__{}__{}".format(args.name,args.env_args["map_name"],args.runner, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard and not args.collect_data:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        if 'map_name' in args.env_args.keys():
            tb_exp_direc = os.path.join(tb_logs_direc,"{}", "{}").format(args.env_args['map_name'],unique_token)
        else:
            tb_exp_direc = os.path.join(tb_logs_direc,"{}", "{}").format(args.env,unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def collect_sequential(args, runner):
    keys = ['state','obs','actions','avail_actions','probs','reward','terminated','filled']
    batch = {}
    start_time=time.time()
    collected_episodes = 0
    data_dir=os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if os.path.exists(data_dir+'/'+args.env_args['map_name']+'_'+args.h5file_suffix + '.h5'):
        os.remove(data_dir+'/'+args.env_args['map_name']+'_'+args.h5file_suffix + '.h5')
    f = h5py.File(data_dir+'/'+args.env_args['map_name']+'_'+args.h5file_suffix + '.h5', 'a')
    
    datasets={}
    while collected_episodes<args.collect_nepisode:
        if args.h5file_suffix=='random':
            batch_tmp = runner.run()
            runner.t_env=0
        else:
            batch_tmp = runner.run(test_mode=True)
        if collected_episodes == 0:
            for key in keys:
                data = batch_tmp[key].to('cpu').numpy()
                max_shape=list(data.shape)
                max_shape[0]=None
                datasets[key]=f.create_dataset(key,maxshape=max_shape,chunks=True,data=data)
                # batch[key] = batch_tmp[key].to('cpu').numpy()
        else:
            for key in keys:
                data = batch_tmp[key].to('cpu').numpy()
                newshape=list(datasets[key].shape)
                newshape[0] += data.shape[0]
                datasets[key].resize(tuple(newshape))
                datasets[key][-data.shape[0]:]=data
                # batch[key] = np.concatenate([batch[key],batch_tmp[key].to('cpu').numpy()],axis=0)
        collected_episodes += runner.batch_size
        if (collected_episodes%(50*runner.batch_size))==0:
            print("Have Collected {} episodes,Time passed: {}".format(collected_episodes,time_str(time.time() - start_time)))
    
    # f = h5py.File(data_dir+'/'+args.env_args['map_name']+'_'+args.h5file_suffix + '.h5', 'w')
    # for key in keys:
    #     f.create_dataset(key, data=batch[key])
    f.close()

    runner.close_env()

def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    if args.use_offline or args.collect_data:
        args.buffer_size=10
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return
    if args.collect_data and args.h5file_suffix != 'medium_replay':
        collect_sequential(args,runner)
        return
    
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    if args.use_offline:
        data_dir=os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets")
        total_datas,hdkey = load_datasets(args,logger,data_dir)
        #####################################
        if getattr(args, 'sparse_lambda', False) or getattr(args, 'cal_dcql', False):
            train_behaviour_policy(args,total_datas,logger,learner,runner,data_dir,hdkey,scheme, groups,preprocess)

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        if args.use_offline:
            sample_number = np.random.choice(len(total_datas[hdkey[0]]), args.batch_size, replace=False)
            off_batch = {}
            for key in hdkey:
                if key !='filled':
                    off_batch[key] = total_datas[key][sample_number].to(args.device)
                else:
                    filled_sample = total_datas[key][sample_number].to(args.device)
            runner.t_env+=int(filled_sample.sum().to('cpu'))
            runner._log(list(off_batch['reward'].sum(1).reshape(-1).to('cpu').numpy()),{"ep_length":filled_sample.sum().to('cpu').float(),"n_episodes":args.batch_size},'')
            new_batch = EpisodeBatch(scheme, groups, args.batch_size, runner.episode_limit + 1,preprocess=preprocess, device=args.device)
            new_batch.update(off_batch)
            new_batch.data.transition_data['filled'] = filled_sample
            if getattr(args, 'cal_dcql', False):
                learner.cal_Dcql(new_batch, runner.t_env)
                exit(0)
            learner.train(new_batch, runner.t_env, episode)
        else:
            with th.no_grad():
                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
                # print('!!!!!!!!!!buffer size:',buffer.episodes_in_buffer,args.buffer_size)
            for _ in range(args.num_circle):
                if buffer.can_sample(args.batch_size):
                    next_episode = episode + args.batch_size_run
                    if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                        continue

                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)
                    del episode_sample

        # Execute test runs once in a while #qy add final test
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # if not args.use_offline:
        if runner.t_env<1000000:
            save_model_interval = 100000
            if args.env=='equal_line':
                save_model_interval=50000
        else:
            save_model_interval = args.save_model_interval
        if args.save_model and (runner.t_env - model_save_time >= save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            if 'map_name' in args.env_args.keys():
                save_path = os.path.join(args.local_results_path, "models", args.env_args['map_name'],args.unique_token, str(runner.t_env))
            else:
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            learner.save_models(save_path)

        episode += args.batch_size_run if args.runner=='parallel' else 1

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    #qy: add final test
    runner.args.test_nepisode = runner.args.test_nepisode*20
    runner.test_returns = []
    runner.test_stats = {}
    n_test_runs = max(1, runner.args.test_nepisode // runner.batch_size)
    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
        time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))

    for _ in range(n_test_runs):
        runner.run(test_mode=True)
    if args.save_model:
        if 'map_name' in args.env_args.keys():
            save_path = os.path.join(args.local_results_path, "models", args.env_args['map_name'],args.unique_token, str(runner.t_env))
        else:
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        # save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)
    if 'madtkd' in args.name:
        if args.teacher and args.save_model:
            save_path = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets",args.env_args['map_name']+'_'+args.h5file_suffix+'_teacher_model')
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
                
    logger.log_stat("episode", episode, runner.t_env)
    logger.print_recent_stats()
    #######################
    if args.h5file_suffix == 'medium_replay' and not args.use_offline:
        sample_medium_replay(args,scheme,buffer)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
