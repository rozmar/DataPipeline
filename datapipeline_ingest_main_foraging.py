import json
project = 'foraging'
with open('dj_local_conf.json') as json_file:
    variables = json.load(json_file)
variables['project'] = project
with open('dj_local_conf.json', 'w') as outfile:
    json.dump(variables, outfile, indent=2, sort_keys=True)
from ingest import datapipeline_metadata
from ingest import datapipeline_behavior
from ingest import datapipeline_elphys
#%%
datapipeline_metadata.populatemetadata()
#datapipeline_behavior.populatebehavior()
#datapipeline_behavior.populatemytables()
#datapipeline_elphys.populateelphys()
#datapipeline_elphys.populatemytables()