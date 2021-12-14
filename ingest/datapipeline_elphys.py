from utils.metaarray import * # to import the very first recording...
import h5py as h5
from utils import configfile
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time as timer
import pandas as pd
import numpy as np
import scipy
import datajoint as dj
import os
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging
import ray

#%%
@ray.remote
def populatemytables_core_paralel(arguments,runround):
    if runround == 1:
        ephysanal.SquarePulse().populate(**arguments)
        ephysanal.SweepFrameTimes().populate(**arguments)
    elif runround == 2:
        ephysanal.SquarePulseSeriesResistance().populate(**arguments)
        ephysanal.SweepSeriesResistance().populate(**arguments)
    elif runround == 3:
        ephysanal.SweepResponseCorrected().populate(**arguments)
    elif runround == 4:
        ephysanal.ActionPotential().populate(**arguments)
        ephysanal.ActionPotentialDetails().populate(**arguments)
        

def populatemytables_core(arguments,runround):
    if runround == 1:
        ephysanal.SquarePulse().populate(**arguments)
        ephysanal.SweepFrameTimes().populate(**arguments)
    elif runround == 2:
        ephysanal.SquarePulseSeriesResistance().populate(**arguments)
        ephysanal.SweepSeriesResistance().populate(**arguments)
        extrapolate_sweep_series_resistance()
    elif runround == 3:
        ephysanal.SweepResponseCorrected().populate(**arguments)
    elif runround == 4:
        ephysanal.ActionPotential().populate(**arguments)
        ephysanal.ActionPotentialDetails().populate(**arguments)
def populatemytables(paralel=True,cores = 6):
    if paralel:
        ray.init(num_cpus = cores)
        for runround in [1,2,3,4]:
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
        for runround in [1,2,3,4]:
            arguments = {'display_progress' : True, 'reserve_jobs' : False}
            populatemytables_core(arguments,runround)

    
    
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
#%% 
def extrapolate_sweep_series_resistance():
    #%
    subject_ids = np.unique((ephysanal.SweepSeriesResistance()&'series_resistance is null').fetch('subject_id'))
    for subject_id in subject_ids:
        print('extrapolating sweep series resistance for subject {}'.format(subject_id))
        cell_numbers = np.unique((ephysanal.SweepSeriesResistance()&'series_resistance is null'&'subject_id = {}'.format(subject_id)).fetch('cell_number'))
        for cell_number in cell_numbers:
            #%
            key = {'subject_id':subject_id,
                   'cell_number':cell_number}
            data_old = pd.DataFrame(ephysanal.SweepSeriesResistance()&key)
            data = data_old.copy()
            nones = data['series_resistance'].values==None
            labels, nums = scipy.ndimage.label(nones)
            for num in range(1,nums+1):
                idxs_now = np.where(labels==num)[0]
                if idxs_now[0]==0:
                    data['series_resistance'][idxs_now]=data['series_resistance'][idxs_now[-1]+1]
                    data['series_resistance_bridged'][idxs_now]=data['series_resistance_bridged'][idxs_now[-1]+1]
                    data['series_resistance_residual'][idxs_now]=data['series_resistance_residual'][idxs_now[-1]+1]
                elif idxs_now[-1]==len(data)-1:
                    data['series_resistance'][idxs_now]=data['series_resistance'][idxs_now[0]-1]
                    data['series_resistance_bridged'][idxs_now]=data['series_resistance_bridged'][idxs_now[0]-1]
                    data['series_resistance_residual'][idxs_now]=data['series_resistance_residual'][idxs_now[0]-1]
                else:
                    rs_start = float(data['series_resistance'][idxs_now[0]-1])
                    rs_end = float(data['series_resistance'][idxs_now[-1]+1])
                    rss = np.arange(rs_start,rs_end,(rs_end-rs_start)/(len(idxs_now)+1))[:-1]
                    data['series_resistance'][idxs_now] = rss
                    data['series_resistance_bridged'][idxs_now] = data['series_resistance_bridged'][idxs_now[0]-1]
                    data['series_resistance_residual'][idxs_now] = data['series_resistance'][idxs_now] - data['series_resistance_bridged'][idxs_now].astype(float)
               #%
            for row_old,row_new in zip(data_old.iterrows(),data.iterrows()):
                row_old = dict(row_old[1])
                row_new = dict(row_new[1] )
                if row_old['series_resistance']  ==  None:
                    del row_old['series_resistance']
                    del row_old['series_resistance_bridged']
                    del row_old['series_resistance_residual']
                    dj.config['safemode'] = False
                    (ephysanal.SweepSeriesResistance()&row_old).delete()
                    dj.config['safemode'] = True  
                    ephysanal.SweepSeriesResistance().insert1(row_new, allow_direct_insert=True)
                    
               
            
                    
#%%
                    
                
            print(cell_number)
    #%%
    
def populateelphys():
  #%%  
    df_subject_wr_sessions=pd.DataFrame(lab.WaterRestriction() * experiment.Session() * experiment.SessionDetails)
    df_subject_ids = pd.DataFrame(lab.Subject())
    if len(df_subject_wr_sessions)>0:
        subject_names = df_subject_wr_sessions['water_restriction_number'].unique()
        subject_names.sort()
    else:
        subject_names = list()
    subject_ids = df_subject_ids['subject_id'].unique()
    #%
    sumdata=list()
    basedir = Path(dj.config['locations.elphysdata_acq4'])
    for setup_dir in basedir.iterdir():
        setup_name=setup_dir.name
        sessions = np.sort(os.listdir(setup_dir))#configfile.readConfigFile(setup_dir.joinpath('.index'))
        for session_acq in sessions[::-1]:#.keys():
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
                                if len(df_subject_wr_sessions) > 0 and len((df_subject_wr_sessions.loc[df_subject_wr_sessions['subject_id'] == subject_id, 'water_restriction_number']).unique())>0:
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
                                        if 'Clamp1.ma' in sweeps.keys():
                                            protocoltype = 'single sweep'
                                            sweepkeys = ['']
                                        else:
                                            protocoltype = 'multiple sweeps'
                                            sweepkeys = sweeps.keys()
                                        for sweep in sweepkeys:
                                            if sweep != '.' and '.txt' not in sweep and '.ma' not in sweep:
                                                sweep_dir = series_dir.joinpath(sweep)    
                                                sweepinfo = configfile.readConfigFile(sweep_dir.joinpath('.index'))
                                                if sweep=='':
                                                    sweep='0'
                                                for file in sweepinfo.keys():
                                                    if '.ma' in file:
                                                        try: # old file version
                                                             
                                                            #print('new file version')
                                                            #%
                                                            ephysfile = h5.File(sweep_dir.joinpath(file), "r")
                                                            data = ephysfile['data'][()]
                                                            metadata_h5 = ephysfile['info']
                                                            metadata = read_h5f_metadata(metadata_h5)
                                                            daqchannels = list(metadata[2]['DAQ'].keys())
                                                            sweepstarttime = datetime.datetime.fromtimestamp(metadata[2]['DAQ'][daqchannels[0]]['startTime'])
                                                            relativetime = (sweepstarttime-cellstarttime).total_seconds()
                                                            if len(ephisdata_cell) > 0 and ephisdata_cell[-1]['sweepstarttime'] == sweepstarttime:
                                                                ephisdata = ephisdata_cell.pop()   
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
                                                                #%
                                                                for idx,channel in enumerate(metadata[0]['cols']):    
                                                                    channelname = channel['name'].decode()
                                                                    if channelname[0] == 'u':
                                                                        channelname = channelname[2:-1]
                                                                        if channelname in ['OrcaFlashExposure','Temperature','LED525','FrameCommand','NextFileTrigger']:
                                                                            ephisdata[channelname] = data[idx]
                                                                            #print('{} added'.format(channelname))
                                                                        else:
                                                                            print('waiting in the other daq')
                                                                            timer.sleep(1000)
                                                            ephisdata_cell.append(ephisdata)
                                                                #%
                                                            
                                                        except: # new file version
                                                            print('old version')
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
                                                           
    # ============================================================================
    #                             if wrname == 'FOR04':
    # =============================================================================
                                # add session to DJ if not present                                   
                                if len(ephisdata_cell)> 0 :
                               
# =============================================================================
#                                     print('waiting')
#                                     timer.sleep(1000)
# =============================================================================
                                    #%
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
                                    #%
                                    if len(ephys_patch.Cell() & celldata) == 0 or len(ephys_patch.Cell()*ephys_patch.Sweep() & celldata)<len(ephisdata_cell):
                                        if len(ephys_patch.Cell()*ephys_patch.Sweep() & celldata)<len(ephisdata_cell):
                                            print('finishing a recording:' )
                                        else:    
                                            print('adding new recording:' )
                                        print(celldata)
                                        if 'type'in serieses['.'].keys():
                                            if serieses['.']['type'] == 'interneuron':
                                                celldata['cell_type'] = 'int'
                                            elif serieses['.']['type'] == 'unknown' or serieses['.']['type'] == 'fail':
                                                celldata['cell_type'] = 'unidentified'
                                            else:
                                                print('unhandled cell type!!')
                                                timer.sleep(1000)
                                        else:
                                            celldata['cell_type'] = 'unidentified'
                                        celldata['cell_recording_start'] = (sweepstarttimes[0]).strftime('%H:%M:%S')
                                        if 'depth' in serieses['.'].keys() and len(serieses['.']['depth'])>0:
                                            celldata['depth'] = int(serieses['.']['depth'])
                                        else:
                                            celldata['depth']= -1
                                        try:
                                            ephys_patch.Cell().insert1(celldata, allow_direct_insert=True)
                                        except dj.errors.DuplicateError:
                                            pass #already uploaded
                                        if 'notes' in serieses['.'].keys():
                                            cellnotes = serieses['.']['notes']
                                        else:
                                            cellnotes = ''
                                        cellnotesdata = {
                                                'subject_id': subject_id,
                                                'session': session,
                                                'cell_number': cell_number,
                                                'notes': cellnotes
                                                }   
                                        try:
                                            ephys_patch.CellNotes().insert1(cellnotesdata, allow_direct_insert=True)
                                        except dj.errors.DuplicateError:
                                            pass #already uploaded
            
                                        #%
                                        for i, ephisdata in enumerate(ephisdata_cell):
                                            
                                                #%
                                            sweep_number = i
                                            print('sweep {}'.format(sweep_number))
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
                                            #%
                                            clampparams_data = ephisdata['metadata'][2]['ClampState']['ClampParams'].copy()
                                            clampparams_data_new = dict()
                                            for clampparamkey in clampparams_data.keys():#6004 is true for some reason.. changing it back to 1
                                                if type(clampparams_data[clampparamkey]) == np.int32:
                                                    if clampparams_data[clampparamkey] > 0:
                                                        clampparams_data[clampparamkey] = int(1)
                                                    else:
                                                        clampparams_data[clampparamkey] = int(0)
                                                else:
                                                    clampparams_data[clampparamkey] = float(clampparams_data[clampparamkey])
                                                clampparams_data_new[clampparamkey.lower()] = clampparams_data[clampparamkey]
                                                #%
                                            sweepmetadata_data = {
                                                    'subject_id': subject_id,
                                                    'session': session,
                                                    'cell_number': cell_number,
                                                    'sweep_number': sweep_number,
                                                    'recording_mode': recording_mode,
                                                    'sample_rate': np.round(1/np.median(np.diff(ephisdata['metadata'][1]['values'])))
                                                    }
                                            sweepmetadata_data.update(clampparams_data_new)
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
                                            #print('waiting')
                                            #timer.sleep(10000)
                                            try:
                                                ephys_patch.Sweep().insert1(sweep_data, allow_direct_insert=True)
                                            except dj.errors.DuplicateError:
                                                pass #already uploaded
                                            try: # maybe it's a duplicate..
                                                ephys_patch.ClampParams().insert1(clampparams_data_new, allow_direct_insert=True)
                                            except dj.errors.DuplicateError:
                                                pass #already uploaded
                                            try:
                                                ephys_patch.SweepMetadata().insert1(sweepmetadata_data, allow_direct_insert=True)
                                            except dj.errors.DuplicateError:
                                                pass #already uploaded
                                            try:
                                                ephys_patch.SweepResponse().insert1(sweepdata_data, allow_direct_insert=True)
                                            except dj.errors.DuplicateError:
                                                pass #already uploaded
                                            try:
                                                ephys_patch.SweepStimulus().insert1(sweepstimulus_data, allow_direct_insert=True)
                                            except dj.errors.DuplicateError:
                                                pass #already uploaded
                                            #%
                                            if 'OrcaFlashExposure' in ephisdata.keys():
                                                sweepimagingexposuredata = {
                                                    'subject_id': subject_id,
                                                    'session': session,
                                                    'cell_number': cell_number,
                                                    'sweep_number': sweep_number,
                                                    'imaging_exposure_trace' :  ephisdata['OrcaFlashExposure']
                                                    }
                                                try:
                                                    ephys_patch.SweepImagingExposure().insert1(sweepimagingexposuredata, allow_direct_insert=True)
                                                except dj.errors.DuplicateError:
                                                    pass #already uploaded
                                            if 'Temperature' in ephisdata.keys():
                                                sweeptemperaturedata = {
                                                    'subject_id': subject_id,
                                                    'session': session,
                                                    'cell_number': cell_number,
                                                    'sweep_number': sweep_number,
                                                    'temperature_trace' : ephisdata['Temperature']*10,
                                                    'temperature_units' :  'degC'
                                                    }
                                                try:
                                                    ephys_patch.SweepTemperature().insert1(sweeptemperaturedata, allow_direct_insert=True)
                                                except dj.errors.DuplicateError:
                                                    pass #already uploaded
                                            if 'LED525' in ephisdata.keys():
                                                sweepLEDdata = {
                                                    'subject_id': subject_id,
                                                    'session': session,
                                                    'cell_number': cell_number,
                                                    'sweep_number': sweep_number,
                                                    'imaging_led_trace' : ephisdata['LED525']}
                                                try:
                                                    ephys_patch.SweepLED().insert1(sweepLEDdata, allow_direct_insert=True)
                                                except dj.errors.DuplicateError:
                                                    pass #already uploaded

# =============================================================================
# 
# #%%
# for subject_id in np.unique((lab.Surgery.VirusInjection()&'virus_id = 240').fetch('subject_id')):
#     dj.config['safemode'] = False
#     #(experiment.Session()&'subject_id = {}'.format(subject_id)).delete()
#     (ephysanal.SquarePulse()&'subject_id = {}'.format(subject_id)).delete()
#     dj.config['safemode'] = True  
# =============================================================================

