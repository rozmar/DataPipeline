import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj
import pandas as pd
from pipeline import pipeline_tools, lab, experiment, behavioranal, ephys_patch, ephysanal
dj.conn()
#%% plot IV
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'same')
    sma[:round(window/2)] = np.nan
    sma[-round(window/2):] = np.nan
    return sma
#%%
def plot_time_interval(wr_name = 'FOR04', cellnum = 1, timeedges = [0, 10],ylimits_response = None, plotRS = False):
    
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]  
    key = {
        'subject_id':subject_id,
        'cell_number':cellnum
    }
    allsweeps=pd.DataFrame((ephys_patch.Sweep()&key))
    sweepstoplot = np.where(np.logical_and(allsweeps['sweep_end_time']>float(np.min(timeedges)) , allsweeps['sweep_start_time']<float(np.max(timeedges))))[0]
    df_iv = pd.DataFrame()
    for sweepnum in sweepstoplot:
        key['sweep_number'] = sweepnum
        df_iv = pd.concat([df_iv,pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepResponse()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))])
    df_IV = pd.DataFrame()
    for line in df_iv.iterrows():
        linenow = line[1]
        time = np.arange(0,len(linenow['response_trace']))/linenow['sample_rate']
        linenow['time'] = time + float(linenow['sweep_start_time'])
        df_IV = pd.concat([df_IV,linenow.to_frame().transpose()])
    fig=plt.figure()
    ax_IV=fig.add_axes([0,0,2,.8])
    ax_stim=fig.add_axes([0,-.6,2,.4])
    for line in df_IV.iterrows():
        ax_IV.plot(line[1]['time'],line[1]['response_trace']*1000,'k-')
        ax_stim.plot(line[1]['time'],line[1]['stimulus_trace']*10**12,'k-')
    ax_IV.set_xlabel('Time (s)')
    ax_IV.set_xlim([np.min(timeedges),np.max(timeedges)])
    if ylimits_response:
        ax_IV.set_ylim([np.min(ylimits_response),np.max(ylimits_response)])
    ax_IV.set_ylabel('mV')
    ax_IV.set_title('Response')      
    
    ax_stim.set_xlabel('Time (s)')
    ax_stim.set_xlim([np.min(timeedges),np.max(timeedges)])
    ax_stim.set_ylabel('pA')
    ax_stim.set_title('Stimulus') 
    if plotRS:
        del key['sweep_number']
        df_RS = pd.DataFrame((ephysanal.SeriesResistance()*ephysanal.SquarePulse())&key)
        needed = (df_RS['square_pulse_start_time'].values>np.min(timeedges)) & (df_RS['square_pulse_start_time'].values<np.max(timeedges))
        ax_RS=fig.add_axes([0,-1.2,2,.4])
        ax_RS.plot(df_RS[needed]['square_pulse_start_time'].values,df_RS[needed]['series_resistance'].values,'ko')
        ax_RS.set_xlabel('Time (s)')
        ax_RS.set_ylabel('RS (MOhm)')
        ax_RS.set_xlim([np.min(timeedges),np.max(timeedges)])
#%%
def plot_IV(wr_name = 'FOR04', cellnum = 1, ivnum = 0,IVsweepstoplot = [0,14]): 
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {
        'subject_id':subject_id,
        'cell_number':cellnum
    }
    sweeps = pd.DataFrame(ephys_patch.Sweep()&key)
    protocolnames = sweeps['protocol_name'].unique()
    ivprotocolnames = [i for i in protocolnames if 'iv' in i.lower()] 
    ivprotocolname = ivprotocolnames[ivnum]
    key['protocol_name'] = ivprotocolname
    #%
    df_iv = pd.DataFrame()
    for sweepnum in IVsweepstoplot:
        key['protocol_sweep_number'] = sweepnum
        df_iv = pd.concat([df_iv,pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepResponse()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))])
    df_IV = pd.DataFrame()
    for line in df_iv.iterrows():
        linenow = line[1]
        time = np.arange(0,len(linenow['response_trace']))/linenow['sample_rate']
        linenow['time'] = time
        df_IV = pd.concat([df_IV,linenow.to_frame().transpose()])
    fig=plt.figure()
    ax_IV=fig.add_axes([0,0,2,.8])
    ax_stim=fig.add_axes([0,-.6,2,.4])
    for line in df_IV.iterrows():
        ax_IV.plot(line[1]['time'],line[1]['response_trace']*1000,'-')
        ax_stim.plot(line[1]['time'],line[1]['stimulus_trace']*10**12,'-')
    ax_IV.set_xlabel('Time (s)')
    ax_IV.set_xlim([0,1])
    
    ax_IV.set_ylabel('mV')
    ax_IV.set_title('Firing pattern')      
    
    ax_stim.set_xlabel('Time (s)')
    ax_stim.set_xlim([0,1])
    ax_stim.set_ylabel('pA')
    ax_stim.set_title('Stimulus')
    
    
#%%
def plot_AP(wr_name = 'FOR04', cellnum = 1, timeedges = [380, 400], timeback = .0015, timeforward = .003, moving_n_diff = 0): 
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]  
    key = {
        'subject_id':subject_id,
        'cell_number':cellnum
    }
    APstoplot = pd.DataFrame(ephysanal.ActionPotential()&key &'ap_max_time >' + str(np.min(timeedges)) &'ap_max_time <' + str(np.max(timeedges)))
    prevsweepnum = np.nan
    Y = list()
    dY = list()
    T = list()
    for ap in APstoplot.iterrows():
        sweepnum = ap[1]['sweep_number']
        if sweepnum != prevsweepnum:
            key['sweep_number'] = sweepnum
            sweepdata = pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepResponse()&key)*(ephys_patch.SweepMetadata()&key))
            sr = sweepdata['sample_rate'].values[0]
            trace = sweepdata['response_trace'].values[0]
            prevsweepnum = sweepnum
        stepback = int(np.round(timeback*sr))
        stepforward = int(np.round(timeforward*sr))
        apmaxidx = ap[1]['ap_max_index']
        if apmaxidx > stepback and apmaxidx<len(trace)-stepforward:
            y = trace[apmaxidx-stepback:apmaxidx+stepforward]
            if moving_n_diff > 1:
                dy = np.diff(movingaverage(y, moving_n_diff))*sr
            else:
                dy = np.diff(y)*sr
            
            dy = np.squeeze(np.asarray(np.nanmean(np.asmatrix([np.concatenate([[np.nan],dy]),np.concatenate([dy,[np.nan]])]),0).transpose()))
            t_y = np.arange(-stepback,stepforward)/sr
            Y.append(y)
            dY.append(dy)
            T.append(t_y)
            #%b
    Y = np.asmatrix(Y).transpose()*1000
    T = np.asmatrix(T).transpose()*1000
    dY = np.asmatrix(dY).transpose()
    #%
    fig=plt.figure()
    ax_v=fig.add_axes([0,0,.8,.8])
    ax_v.plot(T,Y)
    ax_v.set_xlabel('ms')
    ax_v.set_ylabel('mV')
    ax_v.set_xlim([-1*timeback*1000, timeforward*1000])
    ax_dv=fig.add_axes([1,0,.8,.8])
    ax_dv.plot(Y,dY)
    ax_dv.set_xlabel('mV')
    ax_dv.set_ylabel('mV/ms')