from utils.metaarray import * # to import the very first recording...
import h5py as h5
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
import ray

#%
@ray.remote
def populatemytables_core_paralel(arguments,runround):
    if runround == 1:
        ephysanal.SquarePulse().populate(**arguments)
    elif runround == 2:
        ephysanal.SeriesResistance().populate(**arguments)
        ephysanal.ActionPotential().populate(**arguments)

def populatemytables_core(arguments,runround):
    if runround == 1:
        ephysanal.SquarePulse().populate(**arguments)
    elif runround == 2:
        ephysanal.SeriesResistance().populate(**arguments)
        ephysanal.ActionPotential().populate(**arguments)
def populatemytables(paralel=True,cores = 6):
    if paralel:
        ray.init()
        for runround in [1,2]:
            arguments = {'display_progress' : False, 'reserve_jobs' : True,'order' : 'random'}
            print('round '+str(runround)+' of populate')
            result_ids = []
            for coreidx in range(cores):
                result_ids.append(populatemytables_core_paralel.remote(arguments,runround))        
            ray.get(result_ids)
            arguments = {'display_progress' : True, 'reserve_jobs' : False}
            populatemytables_core(arguments,runround)
            
        ray.shutdown()
    else:
        arguments = {'display_progress' : True, 'reserve_jobs' : False,'order' : 'random'}
        populatemytables_core(arguments)
    
    
def read_h5f_metadata(metadata_h5):
    keys_0 = metadata_h5.keys()
    metadata = None
    for key_0 in keys_0:
        if metadata == None:
            if key_0 == '0':
                metadata = list()
            else:
                metadata = dict()
        if type(metadata_h5[key_0]) == h5py._hl.dataset.Dataset:
            datanow = metadata_h5[key_0][()]
        else:
            datanow = read_h5f_metadata(metadata_h5[key_0])
        if type(metadata) == list:
            metadata.append(datanow)
        else:
            metadata[key_0] = datanow
    if len(keys_0) == 0:
        keys_0 = metadata_h5.attrs.keys()
        metadata= dict()
        for key_0 in keys_0:
            if key_0[0]!='_':
                metadata[key_0] = metadata_h5.attrs.get(key_0)
    return metadata
#%%
def populateelphys():
    
    df_subject_wr_sessions=pd.DataFrame(lab.WaterRestriction() * experiment.Session() * experiment.SessionDetails)
    df_subject_ids = pd.DataFrame(lab.Subject())
    subject_names = df_subject_wr_sessions['water_restriction_number'].unique()
    subject_names.sort()
    subject_ids = df_subject_ids['subject_id'].unique()
    #%
    sumdata=list()
    basedir = Path(dj.config['locations.elphysdata_acq4'])
    for setup_dir in basedir.iterdir():
        setup_name=setup_dir.name
        sessions = configfile.readConfigFile(setup_dir.joinpath('.index'))
        for session_acq in sessions.keys():
            if session_acq != '.' and session_acq != 'log.txt':
                session_dir = setup_dir.joinpath(session_acq)
                try:
                    cells = configfile.readConfigFile(session_dir.joinpath('.index'))
                except: # if there is no file
                    cells = None
                if  cells and 'WR_name/ID' in cells['.'].keys(): # it needs to have WRname
                    wrname_ephys = cells['.']['WR_name/ID']
                    wrname = None
                    for wrname_potential in subject_names: # look for water restriction number
                        if wrname_potential.lower() in wrname_ephys.lower():
                            wrname = wrname_potential
                            subject_id = (df_subject_wr_sessions.loc[df_subject_wr_sessions['water_restriction_number'] == wrname, 'subject_id']).unique()[0]
                    if wrname == None: # look for animal identifier:
                        for wrname_potential in subject_ids: # look for water restriction number
                            if str(wrname_potential) in wrname_ephys.lower():
                                subject_id = wrname_potential
                                if len((df_subject_wr_sessions.loc[df_subject_wr_sessions['subject_id'] == subject_id, 'water_restriction_number']).unique())>0:
                                    wrname = (df_subject_wr_sessions.loc[df_subject_wr_sessions['subject_id'] == subject_id, 'water_restriction_number']).unique()[0]
                                else:
                                    wrname = 'no water restriction number for this mouse'
                                
                    if wrname:
                        session_date = (session_acq[0:session_acq.find('_')]).replace('.','-')
                        
                        print('animal: '+ str(subject_id)+'  -  '+wrname)##
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
                                                        try: # old file version
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
                                                        except: # new file version
                                                            #%%
                                                            ephysfile = h5.File(sweep_dir.joinpath(file), "r")
                                                            data = ephysfile['data'][()]
                                                            metadata_h5 = ephysfile['info']
                                                            metadata = read_h5f_metadata(metadata_h5)
                                                            daqchannels = list(metadata[2]['DAQ'].keys())
                                                            sweepstarttime = datetime.datetime.fromtimestamp(metadata[2]['DAQ'][daqchannels[0]]['startTime'])
                                                            relativetime = (sweepstarttime-cellstarttime).total_seconds()
                                                            if len(ephisdata_cell) > 0 and ephisdata_cell[-1]['sweepstarttime'] == sweepstarttime:
                                                                ephisdata = ephisdata_cell[-1]
                                                            else:
                                                                ephisdata = dict()
                                                            if 'primary' in daqchannels: # ephys data
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
                                                            else:# other daq stuff
                                                                #%%
                                                                for idx,channel in enumerate(metadata[0]['cols']):    
                                                                    channelname = channel['name'].decode()
                                                                    if channelname[0] == 'u':
                                                                        channelname = channelname[2:-1]
                                                                        if channelname in ['OrcaFlashExposure','Temperature']:
                                                                            ephisdata[channelname] = data[idx]
                                                                        else:
                                                                            print('waiting in the other daq')
                                                                            timer.sleep(1000)
                                                                    
                                                                        
                                                            #%%
                                                            ephisdata_cell.append(ephisdata)
                                                            #%%
    # =============================================================================
    #                             if wrname == 'FOR04':
    # =============================================================================
                                # add session to DJ if not present
                                if len(ephisdata_cell)> 0 :
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
                                        if 'type'in serieses['.'].keys():
                                            if serieses['.']['type'] == 'interneuron':
                                                celldata['cell_type'] = 'int'
                                            else:
                                                print('unhandled cell type!!')
                                                timer.sleep(1000)
                                        else:
                                            celldata['cell_type'] = 'unidentified'
                                        celldata['cell_recording_start'] = (sweepstarttimes[0]).strftime('%H:%M:%S')
                                        if 'depth' in serieses['.'].keys():
                                            celldata['depth'] = int(serieses['.']['depth'])
                                        else:
                                            celldata['depth']= -1
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
                                            
                                                #%%
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
             
                                            if 'mode' in ephisdata['metadata'][2]['ClampState']:# old file version
                                                recmode = ephisdata['metadata'][2]['ClampState']['mode']
                                            else:
                                                recmode = ephisdata['metadata'][2]['Protocol']['mode']
                                                
                                            if 'IC' in str(recmode):
                                                recording_mode = 'current clamp'
                                            else:
                                                print('unhandled recording mode, please act..')
                                                timer.sleep(10000)
                                            
                                            channelnames = list()
                                            channelunits = list()
                                            for line_now in ephisdata['metadata'][0]['cols']:
                                                if type(line_now['name']) == bytes:
                                                    channelnames.append(line_now['name'].decode().strip("'"))
                                                    channelunits.append(line_now['units'].decode().strip("'"))
                                                else:
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
                                            #%%
                                            if 'OrcaFlashExposure' in ephisdata.keys():
                                                sweepimagingexposuredata = {
                                                    'subject_id': subject_id,
                                                    'session': session,
                                                    'cell_number': cell_number,
                                                    'sweep_number': sweep_number,
                                                    'imaging_exposure_trace' :  ephisdata['OrcaFlashExposure']
                                                    }
                                                ephys_patch.SweepImagingExposure().insert1(sweepimagingexposuredata, allow_direct_insert=True)
                                            if 'Temperature' in ephisdata.keys():
                                                sweeptemperaturedata = {
                                                    'subject_id': subject_id,
                                                    'session': session,
                                                    'cell_number': cell_number,
                                                    'sweep_number': sweep_number,
                                                    'temperature_trace' : ephisdata['Temperature'],
                                                    'temperature_units' :  'dunno'
                                                    }
                                                ephys_patch.SweepTemperature().insert1(sweeptemperaturedata, allow_direct_insert=True)
# =============================================================================
#                                             print('waiting for adding daq')
#                                             timer.sleep(1000)
# =============================================================================
# =============================================================================
#                             else:
#                                 print('waiting')
#                                 timer.sleep(1000)
#         
# =============================================================================
                                                    
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
