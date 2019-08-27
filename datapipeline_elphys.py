from utils.metaarray import * # to import the very first recording...
from utils import configfile
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time as timer
import pandas as pd
import numpy as np
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal

#%
def populatemytables():
    arguments = {'display_progress' : True, 'reserve_jobs' : False,'order' : 'random'}
    ephysanal.SquarePulse().populate(**arguments)
    ephysanal.SeriesResistance().populate(**arguments)
    ephysanal.ActionPotential().populate(**arguments)
#%%
def populateelphys():
    df_subject_wr_sessions=pd.DataFrame(lab.WaterRestriction() * experiment.Session() * experiment.SessionDetails)
    subject_names = df_subject_wr_sessions['water_restriction_number'].unique()
    subject_names.sort()
    #%
    sumdata=list()
    basedir = Path(dj.config['locations.elphysdata_acq4'])
    for setup_dir in basedir.iterdir():
        setup_name=setup_dir.name
        sessions = configfile.readConfigFile(setup_dir.joinpath('.index'))
        for session_acq in sessions.keys():
            if session_acq != '.' and session_acq != 'log.txt':
                session_dir = setup_dir.joinpath(session_acq)
                cells = configfile.readConfigFile(session_dir.joinpath('.index'))
                wrname_ephys = cells['.']['WR_name/ID']
                wrname = None
                for wrname_potential in subject_names:
                    if wrname_potential.lower() in wrname_ephys.lower():
                        wrname = wrname_potential
                if wrname:
                    session_date = (session_acq[0:session_acq.find('_')]).replace('.','-')
                    print('animal: '+wrname)##
                    subject_id = (df_subject_wr_sessions.loc[df_subject_wr_sessions['water_restriction_number'] == wrname, 'subject_id']).unique()[0]
                    if setup_name == 'Voltage_rig_1P':
                        setupname = 'Voltage-Imaging-1p'
                    else:
                        print('unkwnown setup, please add')
                        timer.wait(1000)
                    if 'experimenter' in cells['.'].keys():
                        username = cells['.']['experimenter']
                    else:
                        username = 'rozsam'
                        print('username not specified in acq4 file, assuming rozsam')
                    ### check if session already exists
                    sessiondata = {
                                    'subject_id': subject_id,#(lab.WaterRestriction() & 'water_restriction_number = "'+df_behavior_session['subject'][0]+'"').fetch()[0]['subject_id'],
                                    'session' : np.nan,
                                    'session_date' : session_date,
                                    'session_time' : np.nan,#session_time.strftime('%H:%M:%S'),
                                    'username' : username,
                                    'rig': setupname
                                    }                
                    for cell in cells.keys():
                        if cell != '.' and cell != 'log.txt':
                            ephisdata_cell = list()
                            sweepstarttimes = list()
                            cell_dir =  session_dir.joinpath(cell)
                            serieses = configfile.readConfigFile(cell_dir.joinpath('.index'))
                            cellstarttime = datetime.datetime.fromtimestamp(serieses['.']['__timestamp__'])
                            for series in serieses.keys():
                                if series != '.' and series != 'log.txt':
                                    series_dir =  cell_dir.joinpath(series)
                                    sweeps = configfile.readConfigFile(series_dir.joinpath('.index'))
                                    for sweep in sweeps.keys():
                                        if sweep != '.' and '.txt' not in sweep and '.ma' not in sweep:
                                            sweep_dir = series_dir.joinpath(sweep)    
                                            sweepinfo = configfile.readConfigFile(sweep_dir.joinpath('.index'))
                                            for file in sweepinfo.keys():
                                                if '.ma' in file:
                                                    ephysfile = MetaArray()
                                                    ephysfile.readFile(sweep_dir.joinpath(file))
                                                    data = ephysfile.asarray()
                                                    metadata = ephysfile.infoCopy()
                                                    sweepstarttime = datetime.datetime.fromtimestamp(metadata[2]['startTime'])
                                                    relativetime = (sweepstarttime-cellstarttime).total_seconds()
                                                    ephisdata = dict()
                                                    ephisdata['V']=data[1]
                                                    ephisdata['stim']=data[0]
                                                    ephisdata['data']=data
                                                    ephisdata['metadata']=metadata
                                                    ephisdata['time']=metadata[1]['values']
                                                    ephisdata['relativetime']= relativetime
                                                    ephisdata['sweepstarttime']= sweepstarttime
                                                    ephisdata['series']= series
                                                    ephisdata['sweep']= sweep
                                                    sweepstarttimes.append(sweepstarttime)
                                                    ephisdata_cell.append(ephisdata)
                            # add session to DJ if not present
                            if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"') == 0:
                                if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"') == 0:
                                    sessiondata['session'] = 1
                                else:
                                    sessiondata['session'] = len((experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"').fetch()['session']) + 1
                                sessiondata['session_time'] = (sweepstarttimes[0]).strftime('%H:%M:%S') # the time of the first sweep will be the session time
                                experiment.Session().insert1(sessiondata)
                            #%
                            session = (experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"').fetch('session')[0]
                            cell_number = int(cell[cell.find('_')+1:])
                            #add cell if not added already
                            celldata= {
                                    'subject_id': subject_id,
                                    'session': session,
                                    'cell_number': cell_number,
                                    }
                            if len(ephys_patch.Cell() & celldata) == 0:
                                print('adding new recording:' )
                                print(celldata)
                                if serieses['.']['type'] == 'interneuron':
                                    celldata['cell_type'] = 'int'
                                else:
                                    print('unhandled cell type!!')
                                    timer.sleep(1000)
                                celldata['cell_recording_start'] = (sweepstarttimes[0]).strftime('%H:%M:%S')
                                celldata['depth'] = int(serieses['.']['depth'])
                                ephys_patch.Cell().insert1(celldata, allow_direct_insert=True)
                                cellnotesdata = {
                                        'subject_id': subject_id,
                                        'session': session,
                                        'cell_number': cell_number,
                                        'notes': serieses['.']['notes']
                                        }                            
                                ephys_patch.CellNotes().insert1(cellnotesdata, allow_direct_insert=True)
    
                                #%
                                for i, ephisdata in enumerate(ephisdata_cell):
                                    sweep_number = i
                                    sweep_data = {
                                            'subject_id': subject_id,
                                            'session': session,
                                            'cell_number': cell_number,
                                            'sweep_number': sweep_number,
                                            'sweep_start_time':(ephisdata['sweepstarttime']-sweepstarttimes[0]).total_seconds(),
                                            'sweep_end_time': (ephisdata['sweepstarttime']-sweepstarttimes[0]).total_seconds()+ephisdata['time'][-1],
                                            'protocol_name': ephisdata['series'],#[:ephisdata['series'].find('_')],
                                            'protocol_sweep_number': int(ephisdata['sweep'])
                                            }
     
                                    if ephisdata['metadata'][2]['ClampState']['mode'] == 'IC':
                                        recording_mode = 'current clamp'
                                    else:
                                        print('unhandled recording mode, please act..')
                                        timer.sleep(10000)
                                    
                                    channelnames = list()
                                    channelunits = list()
                                    for line_now in ephisdata['metadata'][0]['cols']:
                                        channelnames.append(line_now['name'])
                                        channelunits.append(line_now['units'])
                                    commandidx = np.where(np.array(channelnames) == 'command')[0][0]
                                    dataidx = np.where(np.array(channelnames) == 'primary')[0][0]
                                    
                                    sweepmetadata_data = {
                                            'subject_id': subject_id,
                                            'session': session,
                                            'cell_number': cell_number,
                                            'sweep_number': sweep_number,
                                            'recording_mode': recording_mode,
                                            'sample_rate': np.round(1/np.median(np.diff(ephisdata['metadata'][1]['values'])))
                                            }
                                    
                                    sweepdata_data = {
                                            'subject_id': subject_id,
                                            'session': session,
                                            'cell_number': cell_number,
                                            'sweep_number': sweep_number,
                                            'response_trace': ephisdata['data'][dataidx,:],
                                            'response_units': ephisdata['metadata'][0]['cols'][dataidx]['units']
                                            }
                                    
                                    sweepstimulus_data = {
                                            'subject_id': subject_id,
                                            'session': session,
                                            'cell_number': cell_number,
                                            'sweep_number': sweep_number,
                                            'stimulus_trace': ephisdata['data'][commandidx,:],
                                            'stimulus_units': ephisdata['metadata'][0]['cols'][commandidx]['units']
                                            }
                                    ephys_patch.Sweep().insert1(sweep_data, allow_direct_insert=True)
                                    ephys_patch.SweepMetadata().insert1(sweepmetadata_data, allow_direct_insert=True)
                                    ephys_patch.SweepResponse().insert1(sweepdata_data, allow_direct_insert=True)
                                    ephys_patch.SweepStimulus().insert1(sweepstimulus_data, allow_direct_insert=True)
    
    
                                                    
# =============================================================================
#      
#  
# #%%
# file = '/home/rozmar/Network/Voltage_imaging_rig_1p_elphys/acq4/rozsam/2019.08.17_000/cell_001/CCIV_001/001/Clamp1.ma'
# ephysfile = MetaArray()
# ephysfile.readFile(file)
# data = ephysfile.asarray()
# metadata = ephysfile.infoCopy()
# 
# 
# 
# 
# 
# import matplotlib.pyplot as plt
# import pyabf
# 
# abf = pyabf.ABF("/home/rozmar/Downloads/CPN trace for Marci.abf")
# abf.setSweep(0)
# #%%
# plt.plot(abf.sweepX, abf.sweepY)
# plt.show()
# plt.plot(abf.sweepX, abf.sweepY)
# plt.xlim([.001,.003])
# plt.ylim([-70,-50])
# plt.show()
# plt.plot(abf.sweepX, abf.sweepY)
# plt.xlim([.0042,.0062])
# plt.ylim([-60,-40])
# plt.show()
# 
# #%%
# import h5py
# import numpy as np
# #f = h5py.File('/home/rozmar/Network/Voltage_imaging_rig_1p_elphys/acq4/rozsam/2019.08.20_000/CCIV_001/000/Clamp1.ma','r')
# f = h5py.File('/home/rozmar/Network/Voltage_imaging_rig_1p_elphys/acq4/rozsam/2019.08.17_000/cell_001/CCIV_001/001/Clamp1.ma','r')
#  
# 
# #%%
# 
# 
# #%%
# data = ma.MetaArray(file='/home/rozmar/Data/acq4/Voltage_rig_1P/2019.08.17_000/cell_001/cont_000/001/Clamp1.ma')
# 
# 
# =============================================================================



#%%
