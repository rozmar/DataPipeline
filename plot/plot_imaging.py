import numpy as np
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
def plot_precision_recall(key_cell,binwidth =  30,frbinwidth = 0.01,firing_rate_window = 3,save_figures = True,xlimits =None):
    #%%
    session_time, cell_recording_start = (experiment.Session()*ephys_patch.Cell()&key_cell).fetch1('session_time','cell_recording_start')
    first_movie_start_time =  np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_start_time'),float))
    first_movie_start_time_real = first_movie_start_time + session_time.total_seconds()
    #%
    fr_kernel = np.ones(int(firing_rate_window/frbinwidth))/(firing_rate_window/frbinwidth)
    first_movie_start_time = np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_start_time'),float))
    ephys_matched_ap_times,ephys_unmatched_ap_times,ophys_matched_ap_times,ophys_unmatched_ap_times = (imaging_gt.GroundTruthROI()&key_cell).fetch('ephys_matched_ap_times','ephys_unmatched_ap_times','ophys_matched_ap_times','ophys_unmatched_ap_times')
    #%
    ephys_matched_ap_times = np.concatenate(ephys_matched_ap_times) - first_movie_start_time 
    ephys_unmatched_ap_times = np.concatenate(ephys_unmatched_ap_times) - first_movie_start_time 
    all_ephys_ap_times = np.concatenate([ephys_matched_ap_times,ephys_unmatched_ap_times])
    ophys_matched_ap_times = np.concatenate(ophys_matched_ap_times) - first_movie_start_time 
    ophys_unmatched_ap_times = np.concatenate(ophys_unmatched_ap_times) - first_movie_start_time 
    all_ophys_ap_times = np.concatenate([ophys_matched_ap_times,ophys_unmatched_ap_times])
    all_times = np.concatenate([ephys_matched_ap_times,ephys_unmatched_ap_times,ophys_matched_ap_times,ophys_unmatched_ap_times])
    maxtime = np.max(all_times)
    #%
    fr_bincenters = np.arange(frbinwidth/2,maxtime+frbinwidth,frbinwidth)
    fr_binedges = np.concatenate([fr_bincenters-frbinwidth/2,[fr_bincenters[-1]+frbinwidth/2]])
    fr_e = np.histogram(all_ephys_ap_times,fr_binedges)[0]/frbinwidth
    fr_e = np.convolve(fr_e, fr_kernel,'same')
    fr_o = np.histogram(all_ophys_ap_times,fr_binedges)[0]/frbinwidth
    fr_o = np.convolve(fr_o, fr_kernel,'same')
    #%
    bincenters = np.arange(binwidth/2,maxtime+binwidth,binwidth)
    binedges = np.concatenate([bincenters-binwidth/2,[bincenters[-1]+binwidth/2]])
    ephys_matched_ap_num_binned,tmp = np.histogram(ephys_matched_ap_times,binedges)
    ephys_unmatched_ap_num_binned,tmp = np.histogram(ephys_unmatched_ap_times,binedges)
    ophys_matched_ap_num_binned,tmp = np.histogram(ophys_matched_ap_times,binedges)
    ophys_unmatched_ap_num_binned,tmp = np.histogram(ophys_unmatched_ap_times,binedges)
    precision_binned = ophys_matched_ap_num_binned/(ophys_matched_ap_num_binned+ophys_unmatched_ap_num_binned)
    recall_binned = ephys_matched_ap_num_binned/(ephys_matched_ap_num_binned+ephys_unmatched_ap_num_binned)
    f1_binned = 2*precision_binned*recall_binned/(precision_binned+recall_binned)
    #%
    if xlimits == None:
        xlimits = [binedges[0],binedges[-1]]
    
    fig=plt.figure()#figsize=(50,0)
    ax_rates = fig.add_axes([0,1.4,2,.3])
    ax_spikes = fig.add_axes([0,1,2,.3])
    ax = fig.add_axes([0,0,2,.8])
    ax_snratio = fig.add_axes([0,-1,2,.8])
    ax_latency = fig.add_axes([0,-2,2,.8])
    ax_latency_hist = fig.add_axes([.5,-3,1,.8])
    
    ax.plot(bincenters,precision_binned,'go-',label = 'precision')    
    ax.plot(bincenters,recall_binned,'ro-',label = 'recall')    
    ax.plot(bincenters,f1_binned,'bo-',label = 'f1 score')    
    ax.set_xlim([binedges[0],binedges[-1]])
    ax.set_ylim([0,1])
    ax.legend()
    ax_spikes.plot(ephys_unmatched_ap_times,np.zeros(len(ephys_unmatched_ap_times)),'r|', ms = 10)
    ax_spikes.plot(all_ephys_ap_times,np.zeros(len(all_ephys_ap_times))+.33,'k|', ms = 10)
    ax_spikes.plot(ophys_unmatched_ap_times,np.ones(len(ophys_unmatched_ap_times)),'r|',ms = 10)
    ax_spikes.plot(all_ophys_ap_times,np.ones(len(all_ophys_ap_times))-.33,'g|', ms = 10)
    ax_spikes.set_yticks([0,.33,.67,1])
    ax_spikes.set_yticklabels(['false negative','ephys','ophys','false positive'])
    ax_spikes.set_ylim([-.2,1.2])
    ax_spikes.set_xlim(xlimits)
    
    t,sn = (ephysanal.ActionPotential()*imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()&key_cell).fetch('ap_max_time','apwave_snratio')
    t= np.asarray(t,float) + cell_recording_start.total_seconds() - first_movie_start_time_real
    ax_snratio.plot(t,sn,'ko')
    ax_snratio.set_ylabel('signal to noise ratio')
    ax_snratio.set_ylim([0,20])
    ax_snratio.set_xlim(xlimits)
    ax_rates.plot(fr_bincenters,fr_e,'k-',label = 'ephys')
    ax_rates.plot(fr_bincenters,fr_o,'g-',label = 'ophys')
    ax_rates.legend()
    ax_rates.set_xlim(xlimits)
    ax_rates.set_ylabel('Firing rate (Hz)')
    ax_rates.set_title('subject_id: %d, cell number: %s' %(key_cell['subject_id'],key_cell['cell_number']))
    ax_latency.plot(ephys_matched_ap_times,1000*(ophys_matched_ap_times-ephys_matched_ap_times),'ko')
    ax_latency.set_ylabel('ephys-ophys spike latency (ms)')
    ax_latency.set_ylim([0,10])
    ax_latency.set_xlabel('time from first movie start (s)')
    ax_latency.set_xlim(xlimits)
    
    ax_latency_hist.hist(1000*(ophys_matched_ap_times-ephys_matched_ap_times),np.arange(-5,15,.1))
    ax_latency_hist.set_xlabel('ephys-ophys spike latency (ms)')
    ax_latency_hist.set_ylabel('matched ap count')
    
    data = list()
    data.append(plot_ephys_ophys_trace(key_cell,ephys_matched_ap_times[0],trace_window = 1,show_e_ap_peaks = True,show_o_ap_peaks = True))
    data.append(plot_ephys_ophys_trace(key_cell,ephys_unmatched_ap_times[0],trace_window = 1,show_e_ap_peaks = True,show_o_ap_peaks = True))
    data.append(plot_ephys_ophys_trace(key_cell,ophys_unmatched_ap_times[0],trace_window = 1,show_e_ap_peaks = True,show_o_ap_peaks = True))
    if save_figures:
        fig.savefig('./figures/{}_cell_{}_roi_type_{}.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type']), bbox_inches = 'tight')
        for data_now,fname in zip(data,['matched','false_negative','false_positive']):
            data_now['figure_handle'].savefig('./figures/{}_cell_{}_roi_type_{}_{}.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type'],fname), bbox_inches = 'tight')
# =============================================================================
#     #%%
#     frametimes_all = (imaging_gt.GroundTruthROI()*imaging.MovieFrameTimes()&key_cell).fetch('frame_times')
#     frame_times_diff = list()
#     frame_times_diff_t = list()
#     for frametime in frametimes_all:
#         frame_times_diff.append(np.diff(frametime))
#         frame_times_diff_t.append(frametime[:-1])
#     fig=plt.figure()
#     ax = fig.add_axes([0,0,2,.3])
#     ax.plot(np.concatenate(frame_times_diff_t),np.concatenate(frame_times_diff)*1000)
#     ax.set_ylim([-5,5])
# =============================================================================
    
    #%%
def plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 1,show_stimulus = False,show_e_ap_peaks = False, show_o_ap_peaks = False):
    #%%
# =============================================================================
#     key_cell = {'session': 1,
#                 'subject_id': 456462,
#                 'cell_number': 3,
#                 'motion_correction_method': 'VolPy',
#                 'roi_type': 'VolPy'}
#     time_to_plot=None
#     trace_window = 100
#     show_stimulus = False
#     show_e_ap_peaks = True
#     show_o_ap_peaks = True
# =============================================================================
    
    
    fig=plt.figure()
    ax_ophys = fig.add_axes([0,0,2,.8])
    ax_ephys = fig.add_axes([0,-1,2,.8])
    if show_stimulus:
        ax_ephys_stim = fig.add_axes([0,-1.5,2,.4])
        
    #%
    session_time, cell_recording_start = (experiment.Session()*ephys_patch.Cell()&key_cell).fetch1('session_time','cell_recording_start')
    first_movie_start_time =  np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_start_time'),float))
    first_movie_start_time_real = first_movie_start_time + session_time.total_seconds()
    if not time_to_plot:
        time_to_plot = trace_window/2#ephys_matched_ap_times[0]
    session_time_to_plot = time_to_plot+first_movie_start_time  # time relative to session start
    cell_time_to_plot= session_time_to_plot + session_time.total_seconds() -cell_recording_start.total_seconds() # time relative to recording start
    #%
    sweep_start_times,sweep_end_times,sweep_nums = (ephys_patch.Sweep()&key_cell).fetch('sweep_start_time','sweep_end_time','sweep_number')
    needed_start_time = cell_time_to_plot - trace_window/2
    needed_end_time = cell_time_to_plot + trace_window/2
    #%
    sweep_nums = sweep_nums[((sweep_start_times > needed_start_time) & (sweep_start_times < needed_end_time)) |
                            ((sweep_end_times > needed_start_time) & (sweep_end_times < needed_end_time)) | 
                            ((sweep_end_times > needed_end_time) & (sweep_start_times < needed_start_time)) ]

    ephys_traces = list()
    ephys_trace_times = list()
    ephys_sweep_start_times = list()
    ephys_traces_stim = list()
    for sweep_num in sweep_nums:
        sweep = ephys_patch.Sweep()&key_cell&'sweep_number = %d' % sweep_num
        trace,sr= (ephys_patch.SweepMetadata()*ephys_patch.SweepResponse()&sweep).fetch1('response_trace','sample_rate')
        trace = trace*1000
        sweep_start_time  = float(sweep.fetch('sweep_start_time')) 
        trace_time = np.arange(len(trace))/sr + sweep_start_time + cell_recording_start.total_seconds() - first_movie_start_time_real
        
        trace_idx = (time_to_plot-trace_window/2 < trace_time) & (time_to_plot+trace_window/2 > trace_time)
                
        ax_ephys.plot(trace_time[trace_idx],trace[trace_idx],'k-')
        
        ephys_traces.append(trace)
        ephys_trace_times.append(trace_time)
        ephys_sweep_start_times.append(sweep_start_time)
        
        if show_e_ap_peaks:

            ap_max_index = (ephysanal.ActionPotential()&sweep).fetch('ap_max_index')
            aptimes = trace_time[np.asarray(ap_max_index,int)]
            apVs = trace[np.asarray(ap_max_index,int)]
            ap_needed = (time_to_plot-trace_window/2 < aptimes) & (time_to_plot+trace_window/2 > aptimes)
            aptimes  = aptimes[ap_needed]
            apVs  = apVs[ap_needed]
            ax_ephys.plot(aptimes,apVs,'ro')

        
        if show_stimulus:
            trace_stim= (ephys_patch.SweepMetadata()*ephys_patch.SweepStimulus()&sweep).fetch1('stimulus_trace')
            trace_stim = trace_stim*10**12
            ax_ephys_stim.plot(trace_time[trace_idx],trace_stim[trace_idx],'k-')
            ephys_traces_stim.append(trace_stim)
            
        
    ephysdata = {'ephys_traces':ephys_traces,'ephys_trace_times':ephys_trace_times}
    if show_stimulus:
        ephysdata['ephys_traces_stimulus'] = ephys_traces_stim

    movie_nums, movie_start_times,movie_frame_rates,movie_frame_nums = ((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_number','movie_start_time','movie_frame_rate','movie_frame_num')
    movie_start_times = np.asarray(movie_start_times, float)
    movie_end_times = np.asarray(movie_start_times, float)+np.asarray(movie_frame_nums, float)/np.asarray(movie_frame_rates, float)
    needed_start_time = session_time_to_plot - trace_window/2
    needed_end_time = session_time_to_plot + trace_window/2

    movie_nums = movie_nums[((movie_start_times >= needed_start_time) & (movie_start_times <= needed_end_time)) |
                            ((movie_end_times >= needed_start_time) & (movie_end_times <= needed_end_time)) | 
                            ((movie_end_times >= needed_end_time) & (movie_start_times <= needed_start_time)) ]
    dffs=list()
    frametimes = list()
    for movie_num in movie_nums:
    #movie_num = movie_nums[(session_time_to_plot>movie_start_times)&(session_time_to_plot<movie_end_times)][0]
        key_movie = key_cell.copy()
        key_movie['movie_number'] = movie_num

        dff,gt_roi_number = ((imaging.ROI()*imaging_gt.GroundTruthROI())&key_movie).fetch1('roi_dff','roi_number')
        dff_all,roi_number_all = (imaging.ROI()&key_movie).fetch('roi_dff','roi_number')
        dff_all =dff_all[roi_number_all!=gt_roi_number]
        frame_times = ((imaging.MovieFrameTimes()*imaging_gt.GroundTruthROI())&key_movie).fetch1('frame_times') + (session_time).total_seconds() - first_movie_start_time_real #modified to cell time
        frame_idx = (time_to_plot-trace_window/2 < frame_times) & (time_to_plot+trace_window/2 > frame_times)
        
        dff_list = [dff]
        dff_list.extend(dff_all)
        prevminval = 0
        for dff_now,alpha_now in zip(dff_list,np.arange(1,1/(len(dff_list)+1),-1/(len(dff_list)+1))):
            dfftoplotnow = dff_now[frame_idx] + prevminval
            ax_ophys.plot(frame_times[frame_idx],dfftoplotnow,'g-',alpha=alpha_now)
            prevminval = np.min(dfftoplotnow) -.005
        #ax_ophys.plot(frame_times[frame_idx],dff[frame_idx],'g-')
        dffs.append(dff)
        frametimes.append(frame_times)
        if show_o_ap_peaks:
            if 'raw' in key_cell['roi_type']:
                apidxes = ((imaging.ROI()*imaging_gt.GroundTruthROI())&key_movie).fetch1('roi_spike_indices')
            else:
                apidxes = ((imaging.ROI()*imaging_gt.GroundTruthROI())&key_movie).fetch1('roi_spike_indices')-1
            oap_times = frame_times[apidxes]
            oap_vals = dff[apidxes]
            oap_needed = (time_to_plot-trace_window/2 < oap_times) & (time_to_plot+trace_window/2 > oap_times)
            oap_times = oap_times[oap_needed]
            oap_vals = oap_vals[oap_needed]
            ax_ophys.plot(oap_times,oap_vals,'ro')

    ophysdata = {'ophys_traces':dffs,'ophys_trace_times':frametimes}
    ax_ophys.autoscale(tight = True)
    
# =============================================================================
#     dfff = np.concatenate(dffs)
#     dfff[np.argmin(dfff)]=0
#     ax_ophys.set_ylim([min(dfff),max(dfff)])
# =============================================================================
    ax_ophys.set_xlim([time_to_plot-trace_window/2,time_to_plot+trace_window/2])
    ax_ophys.set_ylabel('dF/F')
    ax_ophys.spines["top"].set_visible(False)
    ax_ophys.spines["right"].set_visible(False)
    
    ax_ophys.invert_yaxis()
    
    
    ax_ephys.autoscale(tight = True)
    ax_ephys.set_xlim([time_to_plot-trace_window/2,time_to_plot+trace_window/2])
    ax_ephys.set_ylabel('Vm (mV))')
    ax_ephys.spines["top"].set_visible(False)
    ax_ephys.spines["right"].set_visible(False)
    
    if show_stimulus:
       # ax_ephys_stim.autoscale(tight = True)
        ax_ephys_stim.set_xlim([time_to_plot-trace_window/2,time_to_plot+trace_window/2])
        ax_ephys_stim.set_ylabel('Injected current (pA))')
        ax_ephys_stim.set_xlabel('time from first movie start (s)')
        ax_ephys_stim.spines["top"].set_visible(False)
        ax_ephys_stim.spines["right"].set_visible(False)
    else:
        ax_ephys.set_xlabel('time from first movie start (s)')
    outdict = {'ephys':ephysdata,'ophys':ophysdata,'figure_handle':fig}
#%%
    return outdict


def plot_IV(subject_id = 454597, cellnum = 1, ivnum = 0,IVsweepstoplot = None): 
#%
    key = {
        'subject_id':subject_id,
        'cell_number':cellnum
    }
    sweeps = pd.DataFrame(ephys_patch.Sweep()&key)
    protocolnames = sweeps['protocol_name'].unique()
    ivprotocolnames = [i for i in protocolnames if 'iv' in i.lower()] 
    ivprotocolname = ivprotocolnames[ivnum]
    key['protocol_name'] = ivprotocolname
    #
    df_iv = pd.DataFrame()
    #%
    if IVsweepstoplot == None:
        #%
        ivsweepnums = (ephys_patch.Sweep()&key).fetch('protocol_sweep_number')
        for iwsweepnum in ivsweepnums:
            key['protocol_sweep_number'] = iwsweepnum
            apnum = len(ephysanal.ActionPotential()*ephys_patch.Sweep()&key)
            if apnum > 2:
                break
        IVsweepstoplot = [0,iwsweepnum]

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
    ax_IV=fig.add_axes([0,0,1,1])
    ax_stim=fig.add_axes([0,-.4,1,.2])
    for line in df_IV.iterrows():
        ax_IV.plot(line[1]['time'],line[1]['response_trace']*1000,'k-')
        ax_stim.plot(line[1]['time'],line[1]['stimulus_trace']*10**12,'k-')
    ax_IV.set_xlabel('Time (s)')
    ax_IV.set_xlim([0,1])
    
    ax_IV.set_ylabel('mV')
    ax_IV.set_title('subject: {} cell: {}'.format(subject_id,cellnum))      
    
    ax_stim.set_xlabel('Time (s)')
    ax_stim.set_xlim([0,1])
    ax_stim.set_ylabel('pA')
    #ax_stim.set_title('Stimulus')
    return fig

def plot_AP_waveforms(key,AP_tlimits = [-.005,.01],save_image = False):
    #%
    select_high_sn_APs = False
    bin_step = .00001
    bin_size = .00025
    #%
    tau_1_on = .64/1000
    tau_2_on = 4.1/1000
    tau_1_ratio_on =  .61
    tau_1_off = .78/1000
    tau_2_off = 3.9/1000
    tau_1_ratio_off = 55
    #%
    movie_numbers,sweep_numbers,apwavetimes,apwaves,famerates,snratio,apnums,ap_threshold = ((imaging_gt.GroundTruthROI()*imaging.Movie()*imaging_gt.ROIAPWave()*ephysanal.ActionPotentialDetails())&key&'ap_real = 1').fetch('movie_number','sweep_number','apwave_time','apwave_dff','movie_frame_rate','apwave_snratio','ap_num','ap_threshold')
    uniquemovienumbers = np.unique(movie_numbers)
    for movie_number in uniquemovienumbers:
        
        fig=plt.figure()
        ax_ephys=fig.add_axes([0,0,1,1])
        
        ax_raw=fig.add_axes([0,1.1,1,1])
        aps_now = movie_numbers == movie_number
        ax_bin=fig.add_axes([1.3,1.1,1,1])
        ax_e_convolved = fig.add_axes([1.3,0,1,1])
        
        
        ax_bin.set_title('{} ms binning'.format(bin_size*1000))
        if select_high_sn_APs :
            medsn = np.median(snratio[aps_now])
            aps_now = (movie_numbers == movie_number) & (snratio>medsn)
        
        framerate = famerates[aps_now][0]
        apwavetimes_conc = np.concatenate(apwavetimes[aps_now])
        apwaves_conc = np.concatenate(apwaves[aps_now])
        prev_sweep = None
        ephys_vs = list()
        for apwavetime,apwave,sweep_number,ap_num in zip(apwavetimes[aps_now],apwaves[aps_now],sweep_numbers[aps_now],apnums[aps_now]):
            wave_needed_idx = (apwavetime>=AP_tlimits[0]-1/framerate) & (apwavetime<=AP_tlimits[1]+1/framerate)
            ax_raw.plot(apwavetime[wave_needed_idx ]*1000,apwave[wave_needed_idx ])
            if prev_sweep != sweep_number:
                #%
                trace = (ephys_patch.SweepResponse()&key&'sweep_number = {}'.format(sweep_number)).fetch1('response_trace')*1000
                e_sr = (ephys_patch.SweepMetadata()&key&'sweep_number = {}'.format(sweep_number)).fetch1('sample_rate')
                stepback = int(np.abs(np.round(AP_tlimits[0]*e_sr)))
                stepforward = int(np.abs(np.round(AP_tlimits[1]*e_sr)))
                ephys_t = np.arange(-stepback,stepforward)/e_sr * 1000
                prev_sweep = sweep_number
                #%
            apmaxindex = (ephysanal.ActionPotential()&key & 'sweep_number = {}'.format(sweep_number) & 'ap_num = {}'.format(ap_num)).fetch1('ap_max_index')
            ephys_v = trace[apmaxindex-stepback:apmaxindex+stepforward]
            ephys_vs.append(ephys_v)
            ax_ephys.plot(ephys_t,ephys_v)
            
            #break
        #%
        mean_ephys_v = np.mean(np.asarray(ephys_vs),0)

#%
        t = np.arange(0,.01,1/e_sr)
        f_on = tau_1_ratio_on*np.exp(t/tau_1_on) + (1-tau_1_ratio_on)*np.exp(-t/tau_2_on)
        f_off = tau_1_ratio_off*np.exp(t[::-1]/tau_1_off) + (1-tau_1_ratio_off)*np.exp(-t[::-1]/tau_2_off)
        f_on = f_on/np.max(f_on)
        f_off = f_off/np.max(f_off)
        kernel = np.concatenate([f_on,np.zeros(len(f_off))])[::-1]
        kernel  = kernel /sum(kernel )
        
        trace_conv0 = np.convolve(np.concatenate([mean_ephys_v[::-1],mean_ephys_v,mean_ephys_v[::-1]]),kernel,mode = 'same') 
        trace_conv0 = trace_conv0[len(mean_ephys_v):2*len(mean_ephys_v)]
        
        kernel = np.ones(int(np.round(e_sr/framerate)))
        kernel  = kernel /sum(kernel )
        trace_conv = np.convolve(np.concatenate([trace_conv0[::-1],trace_conv0,trace_conv0[::-1]]),kernel,mode = 'same') 
        trace_conv = trace_conv[len(mean_ephys_v):2*len(mean_ephys_v)]

        bin_centers = np.arange(np.min(apwavetime),np.max(apwavetime),bin_step)
        
        bin_mean = list()
        for bin_center in bin_centers:
            bin_mean.append(np.mean(apwaves_conc[(apwavetimes_conc>bin_center-bin_size/2) & (apwavetimes_conc<bin_center+bin_size/2)]))
        ax_bin.plot(bin_centers*1000,np.asarray(bin_mean),'g-')
        ax_bin.invert_yaxis()
        ax_bin.set_xlim(np.asarray(AP_tlimits)*1000)
        ax_raw.invert_yaxis()
        ax_raw.autoscale(tight = True)
        ax_raw.set_xlim(np.asarray(AP_tlimits)*1000)
        ax_raw.set_ylabel('dF/F')
        ax_raw.set_title('subject: {} cell: {} movie: {} apnum: {}'.format(key['subject_id'],key['cell_number'],movie_number,sum(aps_now)))
        ax_ephys.set_xlim(np.asarray(AP_tlimits)*1000)
        ax_ephys.set_xlabel('ms')
        ax_ephys.set_ylabel('mV')
        ax_e_convolved.plot(ephys_t,mean_ephys_v,'k-',label = 'mean')
        ax_e_convolved.plot(ephys_t,trace_conv0,'g--',label = 'convolved mean')
        ax_e_convolved.plot(ephys_t,trace_conv,'g-',label = 'convolved & binned mean')
        ax_e_convolved.legend()
        ax_e_convolved.set_xlim(np.asarray(AP_tlimits)*1000)
        ax_e_convolved.set_xlabel('ms')
        plt.show()
        imaging_gt.ROIEphysCorrelation()
        if save_image:
            fig.savefig('./figures/APwaveforms_subject_{}_cell_{}_movie_{}.png'.format(key['subject_id'],key['cell_number'],movie_number), bbox_inches = 'tight')  
        #break
        

def plot_cell_SN_ratio_APwise(roi_type = 'VolPy',v0_max = -35,holding_min = -600,frame_rate_min =300, frame_rate_max = 800 ,bin_num = 10 ):       
    #%% Show S/N ratios for each AP
    cmap = cm.get_cmap('jet')
# =============================================================================
#     bin_num = 10
#     holding_min = -600 #pA
#     v0_max = -35 #mV
#     roi_type = 'Spikepursuit'#'Spikepursuit'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#
# =============================================================================
    key = {'roi_type':roi_type}
    gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
    cells = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
    snratio = list()
    v0s = list()
    holdings = list()
    rss = list()
    threshs =list()
    mintreshs = list()
    f0s_all = list()
    snratios_all = list()
    peakamplitudes_all = list()
    noise_all = list()
    for cell in cells.iterrows():
        cell = cell[1]
        key_cell = dict(cell)    
        del key_cell['Freq']
        snratios,f0,peakamplitudes,noises = (imaging.Movie()*imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()*ephysanal.ActionPotentialDetails()&key_cell&'ap_real = 1'&'movie_frame_rate > {}'.format(frame_rate_min)&'movie_frame_rate < {}'.format(frame_rate_max)).fetch('apwave_snratio','apwave_f0','apwave_peak_amplitude','apwave_noise')
        #f0 =  (imaging_gt.GroundTruthROI()*imaging.ROI()&key_cell).fetch('roi_f0')
        
        sweep = (imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()&key_cell).fetch('sweep_number')[0]
        thresh = (imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()*ephysanal.ActionPotentialDetails()&key_cell&'ap_real = 1').fetch('ap_threshold')
        trace = (ephys_patch.SweepResponse()*imaging_gt.GroundTruthROI()&key_cell&'sweep_number = {}'.format(sweep)).fetch('response_trace')
        trace =trace[0]
        stimulus = (ephys_patch.SweepStimulus()*imaging_gt.GroundTruthROI()&key_cell&'sweep_number = {}'.format(sweep)).fetch('stimulus_trace')
        stimulus =stimulus[0]
        
        RS = (ephysanal.SweepSeriesResistance()*imaging_gt.GroundTruthROI()&key_cell&'sweep_number = {}'.format(sweep)).fetch('series_resistance')
        RS =RS[0]
        
        medianvoltage = np.median(trace)*1000
        holding = np.median(stimulus)*10**12
        #print(np.mean(snratios[:100]))    
        snratio.append(np.mean(snratios[:50]))
        v0s.append(medianvoltage)
        holdings.append(holding)
        rss.append(RS)
        threshs.append(thresh)
        mintreshs.append(np.min(thresh))
        f0s_all.append(f0)
        snratios_all.append(snratios)
        peakamplitudes_all.append(peakamplitudes)
        noise_all.append(noises)
        #plot_AP_waveforms(key_cell,AP_tlimits)
    
    #%%  for each AP
    
    fig=plt.figure()
    ax_sn_f0 = fig.add_axes([0,0,1,1])
    ax_sn_f0_binned = fig.add_axes([0,-1.2,1,1])
    ax_noise_f0 = fig.add_axes([2.6,0,1,1])
    ax_noise_f0_binned = fig.add_axes([2.6,-1.2,1,1])
    ax_peakampl_f0 = fig.add_axes([1.3,0,1,1])
    ax_peakampl_f0_binned = fig.add_axes([1.3,-1.2,1,1])
    for loopidx, (f0,snratio_now,noise_now,peakampl_now,cell_now) in enumerate(zip(f0s_all,snratios_all,noise_all,peakamplitudes_all,cells.iterrows())):
        if len(f0)>0:
            coloridx = loopidx/len(cells)
            cell_now = cell_now[1]
            label_now = 'Subject:{}'.format(cell_now['subject_id'])+' Cell:{}'.format(cell_now['cell_number'])
            ax_sn_f0.plot(f0,snratio_now,'o',ms=1, color = cmap(coloridx), label= label_now)
            ax_noise_f0.plot(f0,noise_now,'o',ms=1, color = cmap(coloridx), label= label_now)
            ax_peakampl_f0.plot(f0,peakampl_now,'o',ms=1, color = cmap(coloridx), label= label_now)
            lows = np.arange(np.min(f0),np.max(f0),(np.max(f0)-np.min(f0))/(bin_num+1))
            highs = lows + (np.max(f0)-np.min(f0))/(bin_num+1)
            mean_f0 = list()
            sd_f0 = list()
            mean_sn = list()
            sd_sn =list()
            mean_noise = list()
            sd_noise =list()
            mean_ampl = list()
            sd_ampl =list()
            for low,high in zip(lows,highs):
                idx = (f0 >= low) & (f0 < high)
                if len(idx)>10:
                    mean_f0.append(np.mean(f0[idx]))
                    sd_f0.append(np.std(f0[idx]))
                    mean_sn.append(np.mean(snratio_now[idx]))
                    sd_sn.append(np.std(snratio_now[idx]))
                    mean_noise.append(np.mean(noise_now[idx]))
                    sd_noise.append(np.std(noise_now[idx]))
                    mean_ampl.append(np.mean(peakampl_now[idx]))
                    sd_ampl.append(np.std(peakampl_now[idx]))
                    
            ax_sn_f0_binned.errorbar(mean_f0,mean_sn,sd_sn,sd_f0,'o-', color = cmap(coloridx), label= label_now)
            ax_noise_f0_binned.errorbar(mean_f0,mean_noise,sd_noise,sd_f0,'o-', color = cmap(coloridx), label= label_now)
            ax_peakampl_f0_binned.errorbar(mean_f0,mean_ampl,sd_ampl,sd_f0,'o-', color = cmap(coloridx), label= label_now)
                    
    ax_sn_f0.set_xlabel('F0')
    ax_sn_f0.set_ylabel('S/N ratio')
    ax_sn_f0_binned.set_xlabel('F0')
    ax_sn_f0_binned.set_ylabel('S/N ratio')
    
    #ax_sn_f0_binned.legend()
    ax_sn_f0_binned.legend(loc='upper center', bbox_to_anchor=(-.45, 1.5), shadow=True, ncol=1)

    
    
    ax_noise_f0.set_xlabel('F0')
    ax_noise_f0.set_ylabel('Noise (std(dF/F))')
    ax_noise_f0_binned.set_xlabel('F0')
    ax_noise_f0_binned.set_ylabel('Noise (std(dF/F))')
    
    ax_peakampl_f0.set_xlabel('F0')
    ax_peakampl_f0.set_ylabel('Peak amplitude (dF/F)')
    ax_peakampl_f0_binned.set_xlabel('F0')
    ax_peakampl_f0_binned.set_ylabel('Peak amplitude (dF/F)')
    
    #%%
    cells['SN']=snratio
    cells['V0']=v0s
    cells['holding']=holdings
    cells['RS']=np.asarray(rss,float)
    print(cells)
    cells = cells[cells['V0']<v0_max]
    cells = cells[cells['holding']>holding_min]
    print(cells)
    #% S/N ratio histogram
    fig=plt.figure()
    ax_hist = fig.add_axes([0,0,1,1])
    ax_hist.hist(cells['SN'].values)
    ax_hist.set_xlabel('S/N ratio of first 50 spikes')
    ax_hist.set_ylabel('# of cells')
    ax_hist.set_title(roi_type.replace('_',' '))
    ax_hist.set_xlim([0,15])
    #fig.savefig('./figures/SN_hist_{}.png'.format(roi_type), bbox_inches = 'tight')
