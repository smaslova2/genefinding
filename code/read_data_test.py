import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "../../data/cnn/human/chr22.tfrecord"
index_path = None
description = {"seq": "int", "annot": "int"}

def decode_fn(features):
    return features['seq'], features['annot']

dataset = TFRecordDataset(tfrecord_path,
                            index_path=None, 
                            description=description, 
                            transform=decode_fn)


loader = torch.utils.data.DataLoader(dataset, batch_size=32)
data = next(iter(loader))

seq = np.zeros((8, 4, 100))
label = np.zeros((8, 1))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(seq), torch.from_numpy(label))
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)