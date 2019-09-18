
#import logging

import datajoint as dj
#import hashlib
#log = logging.getLogger(__name__)


def get_schema_name(name):
    return 'rozmar_foraging-'+name
# =============================================================================
#     try:
#         return dj.config['custom']['{}_{}.database'.format('rozmar',name)]
#     except KeyError:
#         print('KeyError')
#         if name.startswith('ingest'):
#             prefix = '{}_ingest_'.format(dj.config.get('database.user', 'map'))
#         else:
#             prefix = 'map_v1_'
# 
#     return prefix + name
# =============================================================================
def drop_every_schema():
 #%% Drop everything   
 #%%
    schema = dj.schema('rozmar_foraging-behavior-anal')
    schema.drop(force=True) 
     #%%
    schema = dj.schema('rozmar_foraging-ephys-anal')
    schema.drop(force=True)
    #%%
    schema = dj.schema('rozmar_foraging-ephys_patch')
    schema.drop(force=True)
    #%%
    schema = dj.schema('rozmar_foraging-experiment')
    schema.drop(force=True)
    #%%
    schema = dj.schema('rozmar_foraging-lab')
    schema.drop(force=True) 
    
    

# =============================================================================
# class InsertBuffer(object):
#     '''
#     InsertBuffer: a utility class to help managed chunked inserts
# 
#     Currently requires records do not have prerequisites.
#     '''
#     def __init__(self, rel, chunksz=1, **insert_args):
#         self._rel = rel
#         self._queue = []
#         self._chunksz = chunksz
#         self._insert_args = insert_args
# 
#     def insert1(self, r):
#         self._queue.append(r)
# 
#     def insert(self, recs):
#         self._queue += recs
# 
#     def flush(self, chunksz=None):
#         '''
#         flush the buffer
#         XXX: also get pymysql.err.DataError, etc - catch these or pr datajoint?
#         XXX: optional flush-on-error? hmm..
#         '''
#         qlen = len(self._queue)
#         if chunksz is None:
#             chunksz = self._chunksz
# 
#         if qlen > 0 and qlen % chunksz == 0:
#             try:
#                 self._rel.insert(self._queue, **self._insert_args)
#                 self._queue.clear()
#                 return qlen
#             except dj.DataJointError as e:
#                 raise
# 
#     def __enter__(self):
#         return self
# 
#     def __exit__(self, etype, evalue, etraceback):
#         if etype:
#             raise evalue
#         else:
#             return self.flush(1)
# 
# 
# def dict_to_hash(key):
#     """
# 	Given a dictionary `key`, returns a hash string
#     """
#     hashed = hashlib.md5()
#     for k, v in sorted(key.items()):
#         hashed.update(str(v).encode())
#     return hashed.hexdigest()
# =============================================================================
