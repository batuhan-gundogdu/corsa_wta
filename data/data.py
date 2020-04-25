import os
import numpy as np
from dtw import dtw
import pandas as pd
from random import choice


def load (data_dir):
    
    list_of_files = os.listdir(data_dir)   
    dataset_dict = {}
    list_of_mats = []
    speaker_ids = []
    for line in list_of_files:
        speaker_id = int(line[1:4])
        file_name = os.path.join(data_dir, line)
        x = np.loadtxt(file_name, dtype='float32')
        ids_mat = speaker_id * np.ones(x.shape[0], dtype=int)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        dataset_dict[line] = x
        list_of_mats.append(x)
        speaker_ids.append(ids_mat)
    dataset_merged = np.concatenate(list_of_mats)
    speaker_ids = np.concatenate(speaker_ids)
    return dataset_merged, dataset_dict, speaker_ids


def create_corr_data (opts, dataset_dict, max_length = 150):
    "note: remember to alter the dimensionality if you're not using PLP "
    no_dimensions = 16
    decay_weight = 0.9
    cos = lambda x, y: 1-np.dot(x, y)/np.linalg.norm(x,2)/np.linalg.norm(y,2) 
    segment_info = pd.read_csv(os.path.join(opts['utd_pairs_dir'], 'master_graph.nodes'), delimiter='\s+', header=None)
    
    with open(os.path.join(opts['utd_pairs_dir'], 'master_graph.dedups'),"r") as f:
        all_data=[x.split() for x in f.readlines()]
        segment_groups=[list(map(int,x)) for x in all_data]
    durations = segment_info[2] - segment_info[1]
    segments = np.zeros((len(segment_info), max_length, no_dimensions))
    for i in range(len(segment_info)):
        _temp = dataset_dict[segment_info[0][i]]
        duration = min(segment_info[2][i] - segment_info[1][i], max_length) 
        segments[i,0:duration,:] = _temp[segment_info[1][i]:segment_info[1][i]+duration,0:no_dimensions]
    no_groups = len(segment_groups)
    
    for i in range(no_dimensions):
        segments[:,:,i] = segments[:,:,i] * pow(decay_weight, i)

    no_examples = 0
    for i in range(no_groups): 
        for j in range(len(segment_groups[i])): 
            no_examples += 1

    anchor_samples = np.zeros((no_examples, max_length, no_dimensions))
    pos_samples = np.zeros((no_examples, max_length, no_dimensions))
    sample_weights = np.zeros((no_examples, max_length))

    example_no = 0
    for i in range(no_groups):
        for j in range(len(segment_groups[i])):

            anchor = segments[segment_groups[i][j] - 1, 0:durations[segment_groups[i][j] - 1], :] 
            pos_example_no = choice(np.delete(segment_groups[i], j)) - 1 
            pos_example = segments[pos_example_no,0:durations[pos_example_no],: ]

          
            dist, cost_matrix, acc_cost_matrix, path = dtw(anchor, pos_example, dist=cos)

            len_path = min(len(path[0]), max_length) # in case there's paths longer than RNN_length

            anchor_samples[example_no,0:len_path,:] = anchor[path[0][0:len_path],:]
            pos_samples[example_no,0:len_path,:] = pos_example[path[1][0:len_path],:]

            sample_weights[example_no, 0:len_path] = 1

            example_no += 1
    anchor_samples = np.reshape(anchor_samples, (-1,no_dimensions))
    pos_samples = np.reshape(pos_samples, (-1,no_dimensions))
    sample_weights = np.reshape(sample_weights, (-1,))
    anchor_weights_removed = np.array([anchor_samples[x,:] for x in range(len(sample_weights)) if sample_weights[x] == 1])
    pos_samples_removed = np.array([pos_samples[x,:] for x in range(len(sample_weights)) if sample_weights[x] == 1])
    np.save(os.path.join(opts['utd_pairs_dir'], 'pair1.npy'), anchor_weights_removed)
    np.save(os.path.join(opts['utd_pairs_dir'], 'pair2.npy'),  pos_samples_removed)
    return anchor_weights_removed, pos_samples_removed

