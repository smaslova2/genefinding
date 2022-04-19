from re import S
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import torch
from tfrecord.torch.dataset import TFRecordDataset

from Bio import SeqIO




def load_fasta(fasta_file):
    return SeqIO.to_dict(SeqIO.parse(open(fasta_file), 'fasta'))

# takes DNA sequence, outputs one-hot-encoded matrix with rows A, C, G, T
def seq_encode(sequence):
    l = len(sequence)
    x = np.zeros((4,l),dtype = 'int8')
    for j, i in enumerate(sequence):
        if i == "A" or i == "a":
            x[0][j] = 1
        elif i == "C" or i == "c":
            x[1][j] = 1
        elif i == "G" or i == "g":
            x[2][j] = 1
        elif i == "T" or i == "t":
            x[3][j] = 1
    return x

def subset_sequence(name, seq_len, int_len):
    i=0
    bed =[]
    while i<(seq_len-int_len):
        bed.append([name, i, i+int_len])  
        i+=int(int_len/2) 

    bed = np.stack(bed)
    return bed

def annot_encode(start, stop, annot, features):
    samples = np.zeros((len(features)*2, stop-start), dtype='int8')
    for i, feature in enumerate(features):
        for idx, row in annot[(annot['feature']==feature)].iterrows():
            idx = np.arange(max(start, row['start'])-start, min(stop, row['stop'])-start)
            strand = int(row['strand']=="-")*len(features)
            samples[i+strand, idx] = 1
    samples = samples[...,2500:7500]
    return samples

# Write the records to a file.
def np_to_tfrecord(X, Y, file_writer):

    serialized = tf.train.Example(features=tf.train.Features(feature={
        "seq": tf.train.Feature(int64_list=tf.train.Int64List(value=X.flatten())),
        "annot": tf.train.Feature(int64_list=tf.train.Int64List(value=Y.flatten())),
    })).SerializeToString()
    file_writer.write(serialized)

def create_tfrecords(bed, seqs, annotations, features, output): 
    file_writer= tf.io.TFRecordWriter("%s.tfrecord" % output)

    out_bed_file = open("%s.bed" % output, "w")
    for name, start, stop in tqdm(bed):
        start = int(start)
        stop = int(stop)
        starts_within = ((annotations['start']>start) & (annotations['start']<stop))
        ends_within = ((annotations['stop']>start) & (annotations['stop']<stop))
        annot_subset = annotations[(starts_within | ends_within)]
    
        if len(annot_subset)> 0:
            sample = annot_encode(start, stop, annot_subset, features)
            subseq = seq_encode(seqs[start:stop])
            np_to_tfrecord(subseq, sample, file_writer)
            out_bed_file.write('\t'.join([name, str(start), str(stop)]) + '\n')
            
    file_writer.close()
    out_bed_file.close()

    print("DONE!")

def decode_fn(features):
    return features['seq'].astype(np.float32), features['annot'].astype(np.float32)

def get_dataset(path):
    description = {"seq": "int", "annot": "int"}
    dataset = TFRecordDataset(path,
                            index_path=None,
                            description=description,
                            transform=decode_fn)
    return torch.utils.data.DataLoader(dataset, batch_size=2)
    
'''
# Read the data back out.
def decode_fn(serialized):
    example = tf.io.parse_single_example(
        # Data
        serialized,

        # Schema
        {"seq": tf.io.FixedLenFeature([4, 100000], tf.int64), 
         "annot": tf.io.FixedLenFeature([4, 100000], tf.int64)}
    )

    return example['seq'], example['annot']

def get_dataset(path):
    dataset = tf.data.TFRecordDataset([path]).map(decode_fn)
    
    return dataset
'''




if __name__=="__main__":
    tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
    )   
    bin_length=10000
    annotation_file = "../../data/human/gencode.v39.annotation.gff3"
    output_dir = "../../data/cnn/human/"
    features=['gene', 'exon']
    chrs = ["chr%s" %x for x in range(1, 21)]

    #load annotations
    annotations = pd.read_csv(annotation_file, sep="\t", comment="#", header=None)
    annotations.columns = ["seqid", "source", "feature", "start", "stop", "score", "strand", "phase", "attributes"]
    
    for chr in chrs:
        print("Working on %s" % chr)

        #load sequence
        fasta_file = "../../data/human/hg38/%s.fa" % chr
        fasta = load_fasta(fasta_file)
        seq = fasta[chr].seq

        bed = subset_sequence(chr, len(seq), bin_length) 
        create_tfrecords(bed, seq, annotations[annotations['seqid']==chr], features, output_dir + chr)
    
    #data = get_dataset("../../data/cnn/human/chr22.tfrecord")
    #data = get_dataset("chr22.tfrecord")
