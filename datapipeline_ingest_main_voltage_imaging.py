
import json
project = 'voltage imaging'
with open('dj_local_conf.json') as json_file:
    variables = json.load(json_file)
variables['project'] = project
with open('dj_local_conf.json', 'w') as outfile:
    json.dump(variables, outfile, indent=2, sort_keys=True)
#from ingest import datapipeline_metadata
from ingest import datapipeline_elphys
from ingest import datapipeline_imaging
# =============================================================================
# from ingest import datapipeline_imaging_caiman
# datapipeline_imaging_caiman.populatevolpy()
# =============================================================================
homefolder = '/home/rozmar'#'/nrs/svoboda/rozsam'
#%%
#datapipeline_metadata.populatemetadata()
datapipeline_elphys.populateelphys()
datapipeline_elphys.populatemytables()
#%%
datapipeline_imaging.upload_movie_metadata()
#datapipeline_imaging.populatemytables_imaging() # background value - obsolete
datapipeline_imaging.calculate_exposition_times()
#%%
datapipeline_imaging.save_spikepursuit_pipeline()
datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy',motion_corr = 'VolPy')
#datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy_denoised',motion_corr = 'VolPy2x')
#datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy_raw',motion_corr = 'VolPy') # needs caiman to be installed..
#datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy_denoised_raw',motion_corr = 'VolPy2x')
#datapipeline_imaging.upload_background_subtracted_ROIs(roi_type='SpikePursuit',cores = 8)

#%%

datapipeline_imaging.upload_gt_correlations_apwaves(cores = 4)
datapipeline_imaging.populatemytables_gt(cores = 4)
print('done')