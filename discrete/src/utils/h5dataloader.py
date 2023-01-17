import h5py
from torch.utils import data

class H5Dataset(data.Dataset):
    def __init__(self, path):
        self.file_path = path
        self.data = {}
        # self.keys = keys
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file['actions'])
            self.keys = list(file.keys())
            

    def __getitem__(self, index):
        chosen_data = {}
        if len(self.data.keys())==0:
            f=h5py.File(self.file_path,'r')
            for key in self.keys:
                self.data[key] = f.get(key)
        for key in self.keys:
            chosen_data[key] = self.data[key][index]
        return chosen_data

    def __len__(self):
        return self.dataset_len
