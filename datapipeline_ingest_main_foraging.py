import json
project = 'foraging'
with open('dj_local_conf.json') as json_file:
    variables = json.load(json_file)
variables['project'] = project
with open('dj_local_conf.json', 'w') as outfile:
    json.dump(variables, outfile, indent=2, sort_keys=True)
from ingest import datapipeline_metadata
from ingest import datapipeline_behavior
#%%
datapipeline_metadata.populatemetadata()
# =============================================================================
# datapipeline_behavior.populatebehavior(drop_last_session_for_mice_in_training = False)
# datapipeline_behavior.populatemytables(del_tables = True,cores=8)#del_tables = False,cores=3
# =============================================================================
#datapipeline_elphys.populateelphys()
#datapipeline_elphys.populatemytables()

