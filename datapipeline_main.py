import datapipeline_metadata
import datapipeline_behavior
import datapipeline_elphys
#%%
datapipeline_metadata.populatemetadata()
datapipeline_behavior.populatebehavior()
datapipeline_behavior.populatemytables()
datapipeline_elphys.populateelphys()
datapipeline_elphys.populatemytables()

