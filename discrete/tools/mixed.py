import h5py
import numpy as np
import os
# map_list = ['2s3z','5m_vs_6m','3s_vs_5z','6h_vs_8z']
# h5file_suffix_list = ['medium','medium_replay','expert','mixed']

map_name = '2s3z'
f_medium=h5py.File(map_name+'_medium.h5','r')
f_expert=h5py.File(map_name+'_expert.h5','r')
max_shape=list(f_expert['state'].shape)[0]
idx = np.sort(np.random.choice(np.arange(max_shape), size=2500, replace=False))

if os.path.exists(map_name+'_mixed.h5'):
        os.remove(map_name+'_mixed.h5')
f = h5py.File(map_name+'_mixed.h5','a')
for key in f_medium.keys():
    print(key)
    data = np.concatenate([f_medium[key][idx],f_expert[key][idx]],axis=0)
    
    f.create_dataset(key,data=data)

print(f['state'].shape)
f.close()