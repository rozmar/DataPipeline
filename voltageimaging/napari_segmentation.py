#%%
import napari
from magicgui import magicgui
import pickle as pkl
import numpy as np
import os
from pathlib import Path
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("gui qt5")  
import os
os.chdir('/home/rozmar/Scripts/Python/DataPipeline')
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, imaging

#%% 
sessions_dj = experiment.Session()
sessions_text = list()
sessions_dicts = list()
for session_dj in sessions_dj:
    session_movies_dj = imaging.Movie()&session_dj
    movie_dicts = list()
    movie_texts = list()
    for session_movie_dj in session_movies_dj:
        movie_files_dj = imaging.MovieFile()&session_movie_dj
        movie_files_dicts = list()
        for movie_file_dj in movie_files_dj:
            movie_files_dicts.append(movie_file_dj)
        session_movie_dj['movie_files']=np.asarray(movie_files_dicts.copy())
        movie_dicts.append(session_movie_dj)
        movie_texts.append(session_movie_dj['movie_name'])
    if len(movie_dicts)>0:
        session_dj['movies']=np.asarray(movie_dicts.copy())
        session_dj['movie_names']=np.asarray(movie_texts.copy())
        sessions_dicts.append(session_dj)
        sessions_text.append('anm{} - {}'.format(session_dj['subject_id'],session_dj['session_date']))
        #%
order = np.argsort(sessions_text)
sessions_text=np.asarray(sessions_text)[order]
sessions_dicts=np.asarray(sessions_dicts)[order]


#%%

@magicgui(session={'choices':list(sessions_text)},movie={'choices': list(sessions_dicts[0]['movie_names'])})
def movie_selector(session,movie):
    return session,movie
#%
imageselector = movie_selector.Gui()
#%
def select_session(session=None):
    session_idx = np.where(imageselector.session == sessions_text)[0][0]
    #print(session_idx)
    movies=sessions_dicts[session_idx]['movie_names']
    imageselector.movie_widget.clear()
    imageselector.movie_widget.addItems(np.asarray(movies,str))
    
    for i,z in enumerate(movies):
        imageselector.movie_widget.setItemData(i,z)
    imageselector.movie_changed.disconnect()
    imageselector.movie_changed.connect(load_image)
    load_image()
imageselector.session_changed.connect(select_session) 

def load_image(carry_over_ROI=False,force_saved_ROI = False):
    if not type(carry_over_ROI)== bool:
        carry_over_ROI = False
    if not type(force_saved_ROI)== bool:
        force_saved_ROI = False
    #%
    session_idx = np.where(imageselector.session == sessions_text)[0][0]
    movies=sessions_dicts[session_idx]['movie_names']
    movie_idx = np.where(imageselector.movie == movies)[0][0]
    #%
    #print([session_idx,movie_idx])
    rootdir = dj.config['locations.{}'.format(sessions_dicts[session_idx]['movies'][movie_idx]['movie_files'][0]['movie_file_repository'])]
    movie_dir = sessions_dicts[session_idx]['movies'][movie_idx]['movie_files'][0]['movie_file_directory']
    roi_dir = movie_dir[:movie_dir.find('raw')]+'ROI'+movie_dir[movie_dir.find('raw')+3:]
    movie_dir = movie_dir[:movie_dir.find('raw')]+'VolPy'+movie_dir[movie_dir.find('raw')+3:]
    movie_dir = os.path.join(rootdir,movie_dir)
    roi_dir= os.path.join(rootdir,roi_dir)
    motioncorr_1 = pkl.load(open(os.path.join(movie_dir, 'motion_corr_1.pickle'), 'rb'))
    mean_image = motioncorr_1['templates_rig'][-1]
    viewer.layers[0].data=mean_image
    viewer.layers[0].contrast_limits_range = [1600, 2000]
    # load ROI
    Path(roi_dir).mkdir(parents=True, exist_ok=True)
    roifiles = os.listdir(roi_dir)
    if 'VolPy.npy' in roifiles:
        roi_data=np.load(os.path.join(roi_dir,'VolPy.npy'))
        roi_data_loaded=True
    else:
        roi_data_loaded = False
      
    try:
        #save previous ROIs
        roi_layer=viewer.layers.pop(1)
        if not force_saved_ROI and sum(roi_layer.data.flatten())>0:  # not to overwrite the original ROI
            np.save(os.path.join(roi_layer.metadata['roi_dir'],roi_layer.metadata['roi_file']),roi_layer.data)
    except:
        print('no previous selection layer, nothing to save')
    if carry_over_ROI and mean_image.shape == roi_layer.data.shape and not force_saved_ROI:
        roi_layer = viewer.add_labels(data=roi_layer.data, name='ROIs')#,brush_size=3
    else:    
        if not roi_data_loaded:
            roi_data = np.zeros_like(mean_image)
        roi_layer = viewer.add_labels(data=roi_data, name='ROIs')#,brush_size=3
    roi_layer.metadata = {'session_idx':session_idx,
                          'movie_idx':movie_idx,
                          'session':imageselector.session,
                          'movie':imageselector.movie,
                          'roi_dir':roi_dir,
                          'roi_file':'VolPy.npy'}
 
imageselector.movie_changed.connect(load_image) 
#%%
# =============================================================================
# movie_dir_now='/home/rozmar/Data/Voltage_imaging/VolPy/Voltasge_rig_1P/rozsam/20200725-anm466771/40x_1xtube_7A2'
# motioncorr_1 = pkl.load(open(os.path.join(movie_dir_now, 'motion_corr_1.pickle'), 'rb'))
# mean_image = motioncorr_1['templates_rig'][-1]
# =============================================================================
viewer = napari.view_image(np.random.gamma(np.ones([512,512])),name = 'mean image')#np.random.gamma(np.ones([512,512])) #np.asarray(motioncorr_1['templates_rig'])
@viewer.bind_key('r',overwrite=True)
def reset_ROI(viewer):
    load_image(carry_over_ROI=False,force_saved_ROI = True)
@viewer.bind_key('n',overwrite=True)
def load_next_movie(viewer):
    session_idx = np.where(imageselector.session == sessions_text)[0][0]
    #print(session_idx)
    movies=sessions_dicts[session_idx]['movie_names']
    movie_idx = np.where(imageselector.movie == movies)[0][0]
    if movie_idx<len(movies)-1:
        imageselector.movie_changed.disconnect()
        imageselector.movie_widget.setCurrentIndex(movie_idx+1)
        imageselector.movie_changed.connect(load_image)
        load_image(carry_over_ROI=True,force_saved_ROI = False)   
    elif session_idx<len(sessions_text)-1:
        imageselector.session_widget.setCurrentIndex(session_idx+1)
        
        
#%
viewer.window.add_dock_widget(imageselector)
#%
select_session(imageselector.session)
# =============================================================================
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_image(np.asarray(motioncorr_1['templates_rig']),name = '20200725-anm466771/40x_1xtube_7A2')
#     viewer.add_labels(data=np.zeros_like(motioncorr_1['templates_rig'][0]), name='segmented_cells',brush_size=3)
# =============================================================================
