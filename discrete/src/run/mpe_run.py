import datetime
import os
import pprint
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
import h5py
import numpy as np
from functools import partial
from components.episode_buffer import EpisodeBatch

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
        if getattr(args, 'raw_cql', False):
            args.name = 'raw_'+args.name
            args.sparse_lambda = False
        if getattr(args, 'sparse_lambda', False):
            args.name = 'slmin_'+args.name
        if  getattr(args, 'global_cql_alpha', False):
            tb_suffix += '_global_cql_alpha_'+str(args.global_cql_alpha)+'_'
        elif getattr(args, 'cql_alpha', False):
            tb_suffix += '_cql_alpha_'+str(args.cql_alpha)+'_'

        unique_token = "{}__{}__{}__{}__{}".format(args.name,args.scenario_name,tb_suffix,args.runner, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        unique_token = "{}__{}__{}__{}".format(args.name,args.scenario_name,args.runner, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard and not args.collect_data:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc,"{}", "{}").format(args.scenario_name,unique_token)
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

def load_pretrained_opponents(opponents, scenario_name, filename, num_cooperating_agents, num_opponent_agents, device='gpu'):
    save_dict = th.load(filename, map_location=th.device('cpu')) if not th.cuda.is_available() else th.load(filename)

    if device == 'gpu' and th.cuda.is_available():
        fn = lambda x: x.cuda()
    else:
        fn = lambda x: x.cpu()

    if scenario_name in ['simple_tag', 'simple_world']:
        opponent_params = save_dict['agent_params'][num_cooperating_agents:]
    elif scenario_name in ['simple_adversary', 'simple_crypto']:
        opponent_params = save_dict['agent_params'][:num_opponent_agents]

    for i, params in zip(range(len(opponents)), opponent_params):
        opponents[i].load_params_without_optims(params)

        opponents[i].policy.eval()
        opponents[i].target_policy.eval()
        opponents[i].policy = fn(opponents[i].policy)
        opponents[i].target_policy = fn(opponents[i].target_policy)

    print ('finished loading pretrained opponents from: {}'.format(filename))


def is_opponent(scenario_name, agent_idx, num_cooperating_agents, num_opponent_agents):
    is_opponent_flag = False
    if scenario_name in ['simple_tag', 'simple_world']:
        if agent_idx >= num_cooperating_agents:
            is_opponent_flag = True
    elif scenario_name in ['simple_adversary', 'simple_crypto']:
        if agent_idx < num_opponent_agents:
            is_opponent_flag = True
    return is_opponent_flag


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY['mpe_'+args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

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

    opponents = None
    if args.env == 'pan_mpe_env':
        num_agents, num_cooperating_agents, num_opponent_agents = env_info['agent_num_infos']
        opponent_init_params = []
        for agent_idx in range(num_agents):
            if is_opponent(args.scenario_name, agent_idx, num_cooperating_agents, num_opponent_agents):
                opponent_init_params.append({
                    'num_in_pol': env_info['observation_space'][agent_idx].shape[0], 
                    'num_out_pol': env_info['action_space'][agent_idx].n
                })
        opponents = [DDPGAgent(**params) for params in opponent_init_params]

        pretrained_opponent_model_dir = './src/envs/pan_multiagent-particle-envs/pretrained_models/{}_{}_episode_{}.pt'.format(
            args.scenario_name, 
            args.opponent_algo, 
            args.load_opponent_model_episode
        )
        load_pretrained_opponents(opponents, args.scenario_name, pretrained_opponent_model_dir, num_cooperating_agents, num_opponent_agents)


    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac,opponents=opponents)

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
    if args.collect_data:
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
        ##############load data################
        data_dir=os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets")
        # --------------------------- hdf5 -------------------------------
        import h5py
        from utils.h5dataloader import H5Dataset
        from torch.utils import data
        # hdFile_r = h5py.File(data_dir+'/'+args.scenario_name+'_'+args.h5file_suffix + '.h5', 'r')
        # hdkey = list(hdFile_r.keys())
        # hdFile_r.close()
        dataset_dir = data_dir+'/'+args.scenario_name+'_'+args.h5file_suffix + '.h5'
        dataset = H5Dataset(dataset_dir)
        hdkey = dataset.keys
        sample_datas_batch_size = 5000
        # args.batch_size=1
        dataloader = data.DataLoader(dataset,batch_size=sample_datas_batch_size,shuffle=True)
        logger.console_logger.info("Loading data from  {}, totally {} episodes, sample {} episodes".format(dataset_dir,len(dataset),sample_datas_batch_size))
        total_datas = next(iter(dataloader))
        # hdFile_r = h5py.File(data_dir+'/'+args.scenario_name+'_'+args.h5file_suffix + '.h5', 'r')
        # actions_h = np.array(hdFile_r.get('actions'))
        # hdkey = hdFile_r.keys()
        # if 'probs' in hdkey:
        #     probs_h = np.array(hdFile_r.get('probs'))
        # avail_actions_h = np.array(hdFile_r.get('avail_actions'))
        # filled_h = np.array(hdFile_r.get('filled'))
        # obs_h = np.array(hdFile_r.get('obs'))
        # reward_h = np.array(hdFile_r.get('reward'))
        # state_h = np.array(hdFile_r.get('state'))
        # terminated_h = np.array(hdFile_r.get('terminated'))
        # logger.console_logger.info("Loading data from  {}, totally {} episodes".format(data_dir+'/'+args.env_args['map_name']+'_'+args.h5file_suffix + '.h5',actions_h.shape[0]))
        # hdFile_r.close()
        #####################################
        #####################################
        if getattr(args, 'sparse_lambda', False):
            behaviour_train_steps = 0
            if args.behaviour_checkpoint_path != "":
                behaviour_save_path = args.behaviour_checkpoint_path
                os.makedirs(behaviour_save_path, exist_ok=True)
                logger.console_logger.info("Loading behaviour models from {}".format(behaviour_save_path))
                learner.load_behaviour_model()
            else:
                logger.console_logger.info("Training behaviour models")
                behaviour_last_log_T = 0
                while behaviour_train_steps<1e7:
                    sample_number = np.random.choice(len(total_datas[hdkey[0]]), args.batch_size, replace=False)
                    off_batch = {}
                    for key in hdkey:
                        if key !='filled':
                            off_batch[key] = total_datas[key][sample_number].to(args.device)
                        else:
                            filled_sample = total_datas[key][sample_number].to(args.device)

                    # # Run for a whole episode at a time
                    # sample_number = np.random.choice(len(actions_h), args.batch_size, replace=False)
                    # filled_sample = th.tensor(filled_h[sample_number]).to(args.device)
                    # # max_ep_t_h = filled_sample.sum(1).max(0)[0]
                    # # filled_sample = filled_sample
                    # actions_sample = th.tensor(actions_h[sample_number]).to(args.device)
                    # if 'probs' in hdkey:
                    #     probs_sample = th.tensor(probs_h[sample_number]).to(args.device)
                    # avail_actions_sample = th.tensor(avail_actions_h[sample_number]).to(args.device)
                    # obs_sample = th.tensor(obs_h[sample_number]).to(args.device)
                    # reward_sample = th.tensor(reward_h[sample_number]).to(args.device)
                    # state_sample = th.tensor(state_h[sample_number]).to(args.device)
                    # terminated_sample = th.tensor(terminated_h[sample_number]).to(args.device)

                    # off_batch = {}
                    # off_batch['obs'] = obs_sample
                    # off_batch['reward'] = reward_sample
                    # off_batch['actions'] = actions_sample
                    # if 'probs' in hdkey:
                    #     off_batch['probs'] = probs_sample
                    # off_batch['avail_actions'] = avail_actions_sample
                    # # off_batch['filled'] = filled_sample
                    # off_batch['state'] = state_sample
                    # off_batch['terminated'] = terminated_sample
                    new_batch = EpisodeBatch(scheme, groups, args.batch_size, runner.episode_limit + 1,preprocess=preprocess, device=args.device)
                    new_batch.update(off_batch)
                    new_batch.data.transition_data['filled'] = filled_sample
                    behaviour_train_done,bcloss = learner.train_behaviour(new_batch)
                    behaviour_train_steps +=int(filled_sample.sum().to('cpu'))
                    if (behaviour_train_steps-behaviour_last_log_T)>100000:
                        behaviour_last_log_T=behaviour_train_steps
                        logger.console_logger.info("Behaviour model training loss: {}, training steps: {}".format(bcloss,behaviour_train_steps))
                    if behaviour_train_done:
                        behaviour_save_path = data_dir+'/'+args.scenario_name+'_bcmodel'
                        os.makedirs(behaviour_save_path, exist_ok=True)
                        logger.console_logger.info("Saving behaviour models to {}".format(behaviour_save_path))
                        learner.save_behaviour_model(behaviour_save_path)
                        break

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

            # sample_number = np.random.choice(len(actions_h), args.batch_size, replace=False)
            # filled_sample = th.tensor(filled_h[sample_number]).to(args.device)
            # # max_ep_t_h = filled_sample.sum(1).max(0)[0]
            # # filled_sample = filled_sample
            # actions_sample = th.tensor(actions_h[sample_number]).to(args.device)
            # if 'probs' in hdkey:
            #     probs_sample = th.tensor(probs_h[sample_number]).to(args.device)
            # avail_actions_sample = th.tensor(avail_actions_h[sample_number]).to(args.device)
            # obs_sample = th.tensor(obs_h[sample_number]).to(args.device)
            # reward_sample = th.tensor(reward_h[sample_number]).to(args.device)
            # state_sample = th.tensor(state_h[sample_number]).to(args.device)
            # terminated_sample = th.tensor(terminated_h[sample_number]).to(args.device)

            # off_batch = {}
            # off_batch['obs'] = obs_sample
            # off_batch['reward'] = reward_sample
            # off_batch['actions'] = actions_sample
            # if 'probs' in hdkey:
            #     off_batch['probs'] = probs_sample
            # off_batch['avail_actions'] = avail_actions_sample
            # # off_batch['filled'] = filled_sample
            # off_batch['state'] = state_sample
            # off_batch['terminated'] = terminated_sample
            # # off_batch['batch_size'] = args.batch_size
            # # off_batch['max_seq_length'] = max_ep_t_h

            runner.t_env+=int(filled_sample.sum().to('cpu'))
            runner._log(list(off_batch['reward'].sum(1).reshape(-1).to('cpu').numpy()),{"ep_length":filled_sample.sum().to('cpu').float(),"n_episodes":args.batch_size},'')

            new_batch = EpisodeBatch(scheme, groups, args.batch_size, runner.episode_limit + 1,preprocess=preprocess, device=args.device)
            new_batch.update(off_batch)
            new_batch.data.transition_data['filled'] = filled_sample

            learner.train(new_batch, runner.t_env, episode)
        else:

            with th.no_grad():
                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
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

        if not  args.use_offline:
            if runner.t_env<1000000:
                save_model_interval = 100000
            else:
                save_model_interval = args.save_model_interval
            if args.save_model and (runner.t_env - model_save_time >= save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.scenario_name,args.unique_token, str(runner.t_env))
                # save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                #"results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    #qy: add final test
    runner.args.test_nepisode = runner.args.test_nepisode*20
    n_test_runs = max(1, runner.args.test_nepisode // runner.batch_size)
    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
        time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))

    for _ in range(n_test_runs):
        runner.run(test_mode=True)
    if args.save_model:
        save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)
    logger.log_stat("episode", episode, runner.t_env)
    logger.print_recent_stats()
    #######################
    if args.h5file_suffix == 'medium_replay':
        import h5py
        data_dir=os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets",args.scenario_name+'_medium_replay'+  '.h5')
        keys = list(scheme.keys())+['filled']
        f = h5py.File(data_dir, 'w')
        for key in keys:
            episode_batch = buffer.sample(buffer.episodes_in_buffer)
            f.create_dataset(key, data=episode_batch[key].to('cpu').numpy())
        f.close()

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

############################################## from maddpg-pytorch ##############################################
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import copy
from torch.autograd import Variable

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  
            # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def onehot_from_logits(logits, eps=0.0, dim=1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(th.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return th.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(th.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, dim=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    sampled = sample_gumbel(logits.shape, tens_type=type(logits.data))

    if logits.device != 'cpu':
        sampled = sampled.to(logits.device)

    y = logits + sampled
    
    return F.softmax(y / temperature, dim=dim)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, dim=1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y
    return y


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic=None, hidden_dim=64, lr=0.01):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, constrain_out=True)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, constrain_out=True)
        hard_update(self.target_policy, self.policy)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        
    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if explore:
            action = gumbel_softmax(action, hard=True)
        else:
            action = onehot_from_logits(action)
        return action

    def load_params_without_optims(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer = None
############################################## from maddpg-pytorch ##############################################
