import caffe
import lmdb
import numpy as np

N = 1000
input_dim = 1024
output_dim = 1024

input_data = np.zeros((N,1,1,input_dim),dtype=np.float)
output_data = input_data

env = lmdb.open('ring_lmdb',map_size=8204*10)
env.set_mapsize(1024*1024*1024)
with env.begin(write=True) as txn:
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1
        datum.height = 1
        datum.width = input_dim
        datum.data = input_data[i].tobytes()
        datum.label = output_data[i]
        str_id = '{:08}'.format(i)

        txn.put(str_id.encode('ascii'), datum.SerializeToString())

