import os
import os.path as osp
import numpy as np
dataset =['HalfCheetah-v2', 'simple_spread', 'simple_tag', 'simple_world']
diff = ['random', 'medium-replay', 'medium', 'expert']
seeds = ['seed_0_data', 'seed_1_data', 'seed_2_data', 'seed_3_data', 'seed_4_data']
for da in dataset:
    for di in diff:
        sa = []
        for s in seeds:
            dir = osp.join('datasets', da,di,s)
            dones = np.load(osp.join(dir, 'dones_0.npy'))
            if da == 'HalfCheetah-v2':
                # trajs = sum(dones)
                trajs = len(dones)//25
            else:
                trajs = len(dones)//25
            rews = np.load(osp.join(dir, 'rews_0.npy'))
            ret = sum(rews) / trajs
            sa.append(ret)
            # print(da,di,s,ret)
        print(da,di,sum(sa)/len(sa))
            

