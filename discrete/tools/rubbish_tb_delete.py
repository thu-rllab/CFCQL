from os.path import dirname, abspath
import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

import shutil
pwd = os.path.join(dirname(dirname(abspath(__file__))),'results','tb_logs')
file_list=[]
file_name_list=[]
file_dir_list=[]
def gci(filepath):
#遍历filepath下所有文件，包括子目录
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)            
        if os.path.isdir(fi_d):
            gci(fi_d)                  
        else:
            file_list.append(os.path.join(filepath,fi_d))
            file_name_list.append(fi)
            file_dir_list.append(filepath)
 
#递归遍历/root目录下所有文件
gci(pwd)
# print(file_list)

for ii in range(len(file_list)):
    # if ii>10:break
    # print(ii,file_list[ii],file_name_list[ii])
    if file_name_list[ii][:6]=='events':
        ea = event_accumulator.EventAccumulator(file_list[ii])  # 初始化EventAccumulator对象
        ea.Reload()  # 将事件的内容都导进去
        # key=ea.scalars.Keys()[0]
        max_length=0
        for key in ea.scalars.Keys():
            max_length=max(max_length,len(ea.scalars.Items(key)))
        if max_length<10:
            # print(file_dir_list[ii],len(ea.scalars.Items('ep_length_mean')))
            print('remove',file_dir_list[ii])
            shutil.rmtree(file_dir_list[ii])


