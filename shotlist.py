# @Time : 2022/11/22 15:57 
# @Author : zhongyu 
# @File : shotlist.py
import numpy as np
import json

if __name__ == '__main__':
    dis_train = np.load("./east_shots/DisruptiveTrain_lgbm_new.npy")
    dis_val = np.load("./east_shots/DisruptiveValid_lgbm_new.npy")
    dis_test = np.load("./east_shots/DisruptiveTest_lgbm_new.npy")
    nor_train = np.load("./east_shots/NormalTrain_lgbm_new.npy")
    nor_val = np.load("./east_shots/NormalValid_lgbm_new.npy")
    nor_test = np.load("./east_shots/NormalTest_lgbm_new.npy")
    trainlist = np.concatenate((dis_train, nor_train)).tolist()
    vallist = np.concatenate((dis_val, nor_val)).tolist()
    testlist = np.concatenate((dis_test, nor_test)).tolist()
    list_js = dict()
    list_js['train'] = trainlist
    list_js['val'] = vallist
    list_js['test'] = testlist
    filename = './config/list.json'
    with open(filename, 'w') as file_obj:
        json.dump(list_js, file_obj)
