import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
from tensorflow.keras import backend as K


# Others
from IPython.core.debugger import set_trace
from pathlib import Path

import tensorflow as tf
from tqdm.auto import tqdm
import requests


from tcn import TCN, tcn_full_summary, compiled_tcn
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import mixed_precision
tf.random.set_seed(1234)
seed = 1235
import time

working_dir = Path('.')
DATA1_PATH = Path("./Data1")
DATA_PATH = Path("./Datasets/CWRU")
save_model_path = working_dir / 'Model' 
DE_path = DATA_PATH / '12DriveEndFault'
DE_path1 = DE_path / '1730'
DE_path2 = DE_path / '1750'
DE_path3 = DE_path / '1772'
DE_path4 = DE_path / '1797'

FE_path = DATA_PATH / '12FanEndFault'
FE_path1 = FE_path / '1730'
FE_path2 = FE_path / '1750'
FE_path3 = FE_path / '1772'
FE_path4 = FE_path / '1797'

DE48_path = DATA_PATH / '48DriveEndFault'
DE48_path1 = DE48_path / '1730'
DE48_path2 = DE48_path / '1750'
DE48_path3 = DE48_path / '1772'
DE48_path4 = DE48_path / '1797'

Normal_path = DATA_PATH / 'NormalBaseline'
Normal_path1 = Normal_path / '1730'
Normal_path2 = Normal_path / '1750'
Normal_path3 = Normal_path / '1772'
Normal_path4 = Normal_path / '1797'

Paths = [DE_path1, DE_path2, DE_path3, DE_path4, FE_path1, FE_path2, FE_path3, FE_path4, DE48_path1, DE48_path2,  DE48_path4, Normal_path1, Normal_path2, Normal_path3, Normal_path4]
data_path = Paths

val_path = DATA_PATH / 'ForPred'
Val_path= [val_path]

mirrored_strategy = tf.distribute.MirroredStrategy()

def matfile_to_dft(folder_path, drop=None):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
        drop:
            List of column names to be dropped. 
            Default =  ['BA_time', 'FE_time', 'RPM', 'ans', 'i']
    Return:
        DataFrame with preprocessed data
    '''
    if drop == None:
        drop = ['BA_time', 'RPM', 'ans', 'i']
    dic = matfile_to_dict(folder_path)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df.drop(drop, axis=1, errors='ignore')

def matfile_to_dict(folder_path):

    output_dic = {}
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath, squeeze_me=True)
    return output_dic


def remove_dic_items(dic):
    '''
    Remove redundant data in the dictionary returned by matfile_to_dic inplace.
    '''
    # For each file in the dictionary, delete the redundant key-value pairs
    for _, values in dic.items():
        del values['__header__']
        del values['__version__']    
        del values['__globals__']
        
        
        
def rename_keys(dic):
    '''
    Rename some keys so that they can be loaded into a 
    DataFrame with consistent column names
    '''
    # For each file in the dictionary
    for _,v1 in dic.items():
        # For each key-value pair, rename the following keys 
        for k2,_ in list(v1.items()):
            if 'DE_time' in k2:
                v1['DE_time'] = v1.pop(k2)
            elif 'BA_time' in k2:
                v1['BA_time'] = v1.pop(k2)
            elif 'FE_time' in k2:
                v1['FE_time'] = v1.pop(k2)
            elif 'RPM' in k2:
                v1['RPM'] = v1.pop(k2)
                
def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    #if 'B' in filename:
        #print('B %s' % filename)
        #return 'B'
    #elif 'IR' in filename:
        #print('IR %s' % filename)
        #return 'IR'
    #elif 'OR' in filename:
        #print('OR %s' % filename)
        #return 'OR'
    #elif 'Normal' in filename:
        #print('N %s' % filename)
        #return 'N'
    if 'Inner' in filename:
        #print('IR %s' % filename)
        return 'IR'
    elif 'Outer' in filename:
        #print('OR %s' % filename)
        return 'OR'
    elif 'Normal' in filename:
        #print('N %s' % filename)
        return 'N'
    elif 'Ball' in filename:
        #print('B %s' % filename)
        return 'B'
    
def matfile_to_dft(folder_path, drop=None):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
        drop:
            List of column names to be dropped. 
            Default =  ['BA_time', 'FE_time', 'RPM', 'ans', 'i']
    Return:
        DataFrame with preprocessed data
    '''
    if drop == None:
        drop = ['BA_time', 'RPM', 'ans', 'i']
    dic = matfile_to_dict(folder_path)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df.drop(drop, axis=1, errors='ignore')

def matfile_to_dict(folder_path):

    output_dic = {}
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath, squeeze_me=True)
    return output_dic

def create_framest(data_path, split_perc, segment_length, step_length):
    create_frame_start = time.time()
    if len(data_path) == 1:
        df = matfile_to_dft(data_path[0])
    elif len(data_path) == 2:
        df = matfile_to_df(data_path[0])
        df = df.append(matfile_to_dft(data_path[1]), ignore_index=True)
    elif len(data_path) == 3:
        df = matfile_to_df(data_path[0])
        df = df.append(matfile_to_dft(data_path[1]), ignore_index=True)
        df = df.append(matfile_to_dft(data_path[2]), ignore_index=True)
    else:
        df = matfile_to_dft(data_path[0])
        for path in data_path[1:]:
            df = df.append(matfile_to_dft(path), ignore_index=True)
            
    df_filename = df.pop('filename')
    df.insert(0, 'file', df_filename.map(lambda x: str(x).split('/')[-1]))
    FE_df = pd.DataFrame([df.file, df.FE_time, df.label]).T.rename(columns={'FE_time':'signal'})
    FE_df = FE_df.dropna()
    DE_df = pd.DataFrame([df.file, df.DE_time, df.label]).T.rename(columns={'DE_time':'signal'})
    DE_df = DE_df.dropna()
    df = DE_df.append(FE_df, ignore_index=True)
    df = df.sample(frac=1, random_state=2)
    
    features = df.columns[1:]
    target = 'label'

    test_frame = df.sample(frac = split_perc, random_state=2) # shuffle the dataframe, random_state is a number seed for reproducability
    testframe_id_list = test_frame.index
    train_frame = df.drop(testframe_id_list)

    test_df_output = sig_divide(segment_length, step_length,frame=test_frame)
    train_df_output = sig_divide(segment_length, step_length,frame=train_frame)

    train_df_output = train_df_output.sample(frac=1)
    X_train_frame = train_df_output.drop(target, axis=1).copy()
    y_train_frame = train_df_output[[target]].copy()

    test_frame = test_df_output.sample(frac=1)
    X_test_frame = test_frame.drop(target, axis=1).copy()
    y_test_frame = test_frame[[target]].copy()
    #y_test_frame = y_test_frame.sort_values(by='label', ascending=False)
    
    
    return X_train_frame, y_train_frame, X_test_frame, y_test_frame
    
    
    #return X_test_frame, y_test_frame

def create_data_batcht(data_path, split_perc, segment_length, step_length, b_size):

    
    
    create_frame_start = time.time()
    X_train_frame, y_train_frame, X_test_frame, y_test_frame = create_framest(data_path, split_perc, segment_length, step_length)
    X_train = X_train_frame
    X_test = X_test_frame
    y_train = y_train_frame
    y_test = y_test_frame
    print('class balance of train frame: %s' % y_train['label'].value_counts())
    print('class balance of validation (test) frame: %s' % y_test['label'].value_counts())
    iter_range = [*range(0, len(X_train), 1)]
    X_t, y_t, X_s, y_s = [],[], [],[]
    #print('X_train Length: %s , X_test Length: %s' % (len(X_train),len(X_test)))
    for i in iter_range:
        if i < len(X_test): # might need +1 or -1 to X_test
            
            X_t.append(X_train.iloc[i,:].values)        
            y_t.append(stats.mode(y_train.iloc[i])[0][0])
            X_s.append(X_test.iloc[i,:].values)        
            y_s.append(stats.mode(y_test.iloc[i])[0][0])
        else:   

            X_t.append(X_train.iloc[i,:].values)        
            y_t.append(stats.mode(y_train.iloc[i])[0][0])

    
    #tf.config.list_physical_devices(device_type=None)
    
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    
    y_s1 = 1   
    
    train_dataset = tf.data.Dataset.from_tensor_slices((np.array(X_t).reshape(len(X_train), segment_length, 1), np.asarray(y_t).reshape(-1, 1))).batch(b_size).with_options(options).cache()
    test_dataset = tf.data.Dataset.from_tensor_slices((np.array(X_s).reshape(len(X_test), segment_length, 1), np.asarray(y_s).reshape(-1, 1))).batch(b_size).with_options(options).cache()
    #print('Create Batch Time: %s' % (time.time() - create_frame_start))
    return train_dataset, test_dataset, y_s1


def sig_divide(segment_length, step_length, frame):
    create_frame_start = time.time()
    n_sample_points = []
    splits = []
    split_range = []
    num_seq = []
    for i in range(frame.shape[0]):
        n_sample_points.append(len(frame.iloc[i,1]))
    n_sample_points = np.transpose(n_sample_points)

    nmin1_sample_points = n_sample_points - segment_length
    nmin1_remainder = nmin1_sample_points % step_length
    nmin1_sample_points = nmin1_sample_points - nmin1_remainder
    nminrem_sample_points = n_sample_points - nmin1_remainder
    n_splits = nmin1_sample_points / step_length
    
    for i in n_splits:
        val = int(i+1)
        s_range = range(0, val, 1)
        split_range.append(s_range)


    num_seq += range(0, len(n_splits), 1)
    
    dic = {}
    idx = 0
    for i in num_seq:
        split_range_list = split_range[i]
        win_start1 = np.multiply(step_length, split_range_list)
        win_end1 = np.add(segment_length, np.multiply(step_length, split_range_list))
        for segment in list(split_range_list):
            dic[idx] = {
                'signal': frame.iloc[i,1][win_start1[segment]:win_end1[segment]], # [0 + (step_length * (segment - 1)):segment_length + (step_length * segment)], 
                'label': frame.iloc[i,2],
            }
            
            idx += 1
    df_tmp1 = pd.DataFrame.from_dict(dic,orient='index')
    df_output1 = pd.concat([df_tmp1[['label']], pd.DataFrame(np.vstack(df_tmp1["signal"].values)).astype('float32')], axis=1 )
    map_label = {'N':0, 'B':1, 'IR':2, 'OR':3}
    df_output1['label'] = df_output1['label'].map(map_label).astype('int16')
    print('Sig Divide Time: %s' % (time.time() - create_frame_start))
    return df_output1
    
    
    
def create_pred_batch(Val_path, segment_length=2400, step_length=300, b_size=256):
    Vdf = matfile_to_dft(Val_path[0])
    for valpath in Val_path[1:]: #Index for Posix returning indexed tuple of path
        Vdf = Vdf.append(matfile_to_dft(valpath), ignore_index=True)
            
    vdf_filename = Vdf.pop('filename')
    Vdf.insert(0, 'file', vdf_filename.map(lambda v: str(v).split('/')[-1]))
    FE_vdf = pd.DataFrame([Vdf.file, Vdf.FE_time, Vdf.label]).T.rename(columns={'FE_time':'signal'})
    FE_vdf = FE_vdf.dropna()
    DE_vdf = pd.DataFrame([Vdf.file, Vdf.DE_time, Vdf.label]).T.rename(columns={'DE_time':'signal'})
    DE_vdf = DE_vdf.dropna()
    Vdf = DE_vdf.append(FE_vdf, ignore_index=True)
    Vdf = Vdf.sample(frac=1, random_state=2)
    
    val_features = Vdf.columns[1:]
    target_pred = 'label'
	  #test_frame becomes val frame
    val_sig = Vdf.sample(frac = 1, random_state=2) # shuffle the dataframe, random_state is a number seed for reproducability
    #testframe_id_list = test_frame.index
    
    val_df_output = sig_divide(segment_length, step_length, val_sig)

    val_frame = val_df_output.sample(frac=1)
    X_val_frame = val_frame.drop(target_pred, axis=1).copy()
    y_val_frame = val_frame[[target_pred]].copy()
	  # End of create frame	
	
	
    # Create frame returns here


    
    X_val = X_val_frame
    y_val = y_val_frame
    #print('class balance of train frame: %s' % y_train['label'].value_counts())
    #print('class balance of validation (test) frame: %s' % y_test['label'].value_counts())
    iter_range = [*range(0, len(X_val), 1)]
    X_v, y_v = [],[]
    #print('X_train Length: %s , X_test Length: %s' % (len(X_train),len(X_test)))
    for i in iter_range:
        X_v.append(X_val.iloc[i,:].values)        
        y_v.append(stats.mode(y_val.iloc[i])[0][0])

    
    #tf.config.list_physical_devices(device_type=None)
    
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    
    y_v1 = np.asarray(y_v).reshape(-1, 1)
    X_v1 = np.asarray(X_v).reshape(len(X_val), segment_length, 1) #Might just be np.array()
    val_dataset = tf.data.Dataset.from_tensor_slices((X_v1, y_v1)).batch(b_size).with_options(options)
    
    return val_dataset, X_v1, y_v1

