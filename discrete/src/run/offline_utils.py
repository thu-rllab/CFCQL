import os
import numpy as np
from os.path import dirname, abspath
from components.episode_buffer import EpisodeBatch
import h5py
from utils.h5dataloader import H5Dataset
from torch.utils import data
import torch as th
def train_behaviour_policy(args,total_datas,logger,learner,runner,data_dir,hdkey,scheme, groups,preprocess):
    behaviour_train_steps = 0
    if 'map_name' in args.env_args.keys():
        behaviour_checkpoint_path = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets", args.env_args['map_name']+'_'+args.h5file_suffix+"_bcmodel")
    else:
        behaviour_checkpoint_path = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets", args.env+args.h5file_suffix+"bcmodel")

    if os.path.exists(behaviour_checkpoint_path):
        # behaviour_save_path = args.behaviour_checkpoint_path
        logger.console_logger.info("Loading behaviour models from {}".format(behaviour_checkpoint_path))
        learner.load_behaviour_model(behaviour_checkpoint_path)
    else:
        logger.console_logger.info("Training behaviour models")
        behaviour_last_log_T = 0
        while behaviour_train_steps<args.max_behaviour_train_steps:
            # Run for a whole episode at a time
            sample_number = np.random.choice(len(total_datas[hdkey[0]]), args.batch_size, replace=False)
            off_batch = {}
            for key in hdkey:
                if key !='filled':
                    off_batch[key] = total_datas[key][sample_number].to(args.device)
                else:
                    filled_sample = total_datas[key][sample_number].to(args.device)
            new_batch = EpisodeBatch(scheme, groups, args.batch_size, runner.episode_limit + 1,preprocess=preprocess, device=args.device)
            new_batch.update(off_batch)
            new_batch.data.transition_data['filled'] = filled_sample
            behaviour_train_done,bcloss = learner.train_behaviour(new_batch)
            behaviour_train_steps +=int(filled_sample.sum().to('cpu'))
        if 'map_name' in args.env_args.keys():
            behaviour_save_path = data_dir+'/'+args.env_args['map_name']+'_'+args.h5file_suffix+'_bcmodel'
        else:
            behaviour_save_path = data_dir+'/'+args.env+'_'+args.h5file_suffix+'_bcmodel'
        os.makedirs(behaviour_save_path, exist_ok=True)
        logger.console_logger.info("Saving behaviour models to {}".format(behaviour_save_path))
        learner.save_behaviour_model(behaviour_save_path)

def load_datasets(args,logger,data_dir):
    
    # --------------------------- hdf5 -------------------------------
    
    dataset_dir = data_dir+'/'+args.env_args['map_name']+'_'+args.h5file_suffix + '.h5'
    dataset = H5Dataset(dataset_dir)
    hdkey = dataset.keys
    if getattr(args,"training_episodes",False):
        sample_datas_batch_size = args.training_episodes
    else:
        sample_datas_batch_size = 5000

    args.batch_size=min(args.batch_size,sample_datas_batch_size)
    random_idx = np.sort(np.random.choice(len(dataset), min(sample_datas_batch_size,len(dataset)), replace=False))
    total_datas = dataset[random_idx]
    for key in total_datas.keys():
        total_datas[key] = th.tensor(total_datas[key])
    logger.console_logger.info("Loading data from  {}, totally {} episodes, sample {} episodes".format(dataset_dir,len(dataset),sample_datas_batch_size))
    return total_datas,hdkey

def sample_medium_replay(args,scheme,buffer):
    if not os.path.exists(os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets")):
        os.makedirs(os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets"), exist_ok=True)
    data_dir=os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets",args.env_args['map_name']+'_medium_replay'+  '.h5')
    if os.path.exists(data_dir):
        os.remove(data_dir)
    keys = list(scheme.keys())+['filled']
    f = h5py.File(data_dir, 'w')
    for key in keys:
        print('medium_replay:',buffer.episodes_in_buffer)
        episode_batch = buffer.sample(buffer.episodes_in_buffer)
        f.create_dataset(key, data=episode_batch[key].to('cpu').numpy())
    f.close()


