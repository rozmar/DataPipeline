import os
os.chdir('/home/rozmar/Scripts/Python/DataPipeline')
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import datajoint as dj

dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
import plot.plot_imaging
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import time
from pathlib import Path
import json
import decimal
import shutil

font = {'size'   : 15}
matplotlib.rc('font', **font)

def moving_average(a, n=3) : # moving average 
    if n>2:
        begn = int(np.ceil(n/2))
        endn = int(n-begn)-1
        a = np.concatenate([a[begn::-1],a,a[:-endn:-1]])
    ret = np.cumsum(a,axis = 0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

junction_potential = 13.5 #mV
downsampled_rate = 10000 #Hz
#%%
overwrite = False
gt_package_directory = '/home/rozmar/Data/Voltage_imaging/ground_truth'
subject_ids = np.unique(imaging_gt.CellMovieCorrespondance().fetch('subject_id'))
key_list = list()
for subject_id in subject_ids:
    key = {'subject_id' : subject_id}
    cell_numbers = np.unique((imaging_gt.CellMovieCorrespondance()&key).fetch('cell_number'))
    for cell_number in cell_numbers:
        key_now = key.copy()
        key_now['cell_number'] = cell_number
        key_list.append(key_now)
for key in key_list:
    movies = imaging_gt.CellMovieCorrespondance()&key
    save_this_movie = True
    for movie in movies:
        print(movie)
        if len(imaging_gt.GroundTruthROI()&movie&'roi_type = "VolPy"') == 0:
            print('no groundtruth.. skipped')
        else:
            movie_dict = dict((imaging.Movie()&movie).fetch1())
            for keynow in movie_dict.keys():
                if type(movie_dict[keynow]) == decimal.Decimal:
                    movie_dict[keynow] = float(movie_dict[keynow])
            save_dir_base = os.path.join(gt_package_directory,str(key['subject_id']),'Cell_{}'.format(key['cell_number']),movie_dict['movie_name'])
            if ((os.path.isdir(save_dir_base) and (len(os.listdir(save_dir_base))>3 and not overwrite))):
                print('already exported, skipped')
            else:
                sweep_numbers = movie['sweep_numbers']
                del movie['sweep_numbers']
                session_time, cell_recording_start = (experiment.Session()*ephys_patch.Cell()&key).fetch1('session_time','cell_recording_start')
                first_movie_start_time =  np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key).fetch('movie_start_time'),float))
                first_movie_start_time_real = first_movie_start_time + session_time.total_seconds()
                frame_times = (imaging.MovieFrameTimes()&movie).fetch1('frame_times') - cell_recording_start.total_seconds() + session_time.total_seconds()
                movie_dict['movie_start_time'] = frame_times[0]        
                movie_files = list()
                repositories , directories , fnames = (imaging.MovieFile() & movie).fetch('movie_file_repository','movie_file_directory','movie_file_name')
                for repository,directory,fname in zip(repositories,directories,fnames):
                    movie_files.append(os.path.join(dj.config['locations.{}'.format(repository)],directory,fname))
                sweepdata_out = list()
                sweepmetadata_out = list()
                for sweep_number in sweep_numbers:
                    #%
                    key_sweep = key.copy()
                    key_sweep['sweep_number'] = sweep_number
        
                    neutralizationenable,e_sr= (ephys_patch.SweepMetadata()&key_sweep).fetch1('neutralizationenable','sample_rate')
                    try:
                        uncompensatedRS =  float((ephysanal.SweepSeriesResistance()&key_sweep).fetch1('series_resistance_residual'))
                    except:
                        uncompensatedRS = 0
                    v = (ephys_patch.SweepResponse()&key_sweep).fetch1('response_trace')
                    i = (ephys_patch.SweepStimulus()&key_sweep).fetch1('stimulus_trace')
                    tau_1_on =.1/1000
                    t = np.arange(0,.001,1/e_sr)
                    f_on = np.exp(t/tau_1_on) 
                    f_on = f_on/np.max(f_on)
                    kernel = np.concatenate([f_on,np.zeros(len(t))])[::-1]
                    kernel  = kernel /sum(kernel )  
                    i_conv = np.convolve(i,kernel,'same')
                    v_comp = (v - i*uncompensatedRS*10**6)*1000 - junction_potential
                    i = i * 10**12
                    
                    sweep_start_time  = float((ephys_patch.Sweep()&key_sweep).fetch('sweep_start_time')) 
                    trace_t = np.arange(len(v))/e_sr + sweep_start_time
                    
                    downsample_factor = int(np.round(e_sr/downsampled_rate))
                    #%downsampling
                    v_out = moving_average(v_comp, n=downsample_factor)
                    v_out = v_out[int(downsample_factor/2)::downsample_factor]
                    i_out = moving_average(i, n=downsample_factor)
                    i_out = i_out[int(downsample_factor/2)::downsample_factor]
                    t_out = moving_average(trace_t, n=downsample_factor)
                    t_out = t_out[int(downsample_factor/2)::downsample_factor]
                    
                    sweepdata_out.append(np.asarray([t_out,v_out,i_out]))
                    sweep_dict = dict((ephys_patch.SweepMetadata()*ephys_patch.Sweep()&key_sweep).fetch1())
                    for keynow in sweep_dict.keys():
                        if type(sweep_dict[keynow]) == decimal.Decimal:
                            sweep_dict[keynow] = float(sweep_dict[keynow])
                        
                    sweep_dict['sample_rate']=downsampled_rate
                    sweep_dict['ljp'] = junction_potential
                    sweep_dict['ljp_corrected'] = True
                    sweep_dict['data_header'] = ['time','voltage','stimulus']
                    sweep_dict['data_units'] =['s','mV','pA']
                    sweepmetadata_out.append(sweep_dict.copy())
                    #%
    # =============================================================================
    #                 if neutralizationenable == 0:
    #                     save_this_movie = False
    #                     print('no neutralization!!! not saved.')
    #                     #print(key_sweep)
    # =============================================================================
                #break
                if save_this_movie:
                    save_dir_base = os.path.join(gt_package_directory,str(key['subject_id']),'Cell_{}'.format(key['cell_number']),movie_dict['movie_name'])
                    save_dir_movie = os.path.join(save_dir_base,'movie')
                    save_dir_ephys = os.path.join(save_dir_base,'ephys')
                    Path(save_dir_base).mkdir(parents=True, exist_ok=True)
                    Path(save_dir_ephys).mkdir(parents=True, exist_ok=True)
                    Path(save_dir_movie).mkdir(parents=True, exist_ok=True)
                    for ephys_data,ephys_metadata,sweep_number in zip(sweepdata_out,sweepmetadata_out,sweep_numbers):
                        #np.save(os.path.join(save_dir_ephys,'sweep_{}.npy'.format(sweep_number)),ephys_data)
                        np.savez_compressed(os.path.join(save_dir_ephys,'sweep_{}.npz'.format(sweep_number)),
                                            time=ephys_data[0],
                                            voltage = ephys_data[1],
                                            stimulus = ephys_data[2],
                                            units = np.asarray(ephys_metadata['data_units']))
                        with open(os.path.join(save_dir_ephys,'sweep_{}.json'.format(sweep_number)), 'w') as outfile:
                            json.dump(ephys_metadata, outfile, indent=2, sort_keys=True)
                    for movie_file in movie_files:
                        shutil.copyfile(movie_file,os.path.join(save_dir_movie,Path(movie_file).name))
                        #%
                    with open(os.path.join(save_dir_base,'movie_metadata.json'.format(sweep_number)), 'w') as outfile:
                        json.dump(movie_dict, outfile, indent=2, sort_keys=True)
                        #%
                    np.save(os.path.join(save_dir_base,'frame_times.npy'), frame_times)

# =============================================================================
#         snratios = (imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()*ephysanal.ActionPotentialDetails()&movie&'ap_real = 1').fetch('apwave_snratio')
#         imaging_gt.GroundTruthROI()&movie&'roi_type = "SpikePursuit"'
# =============================================================================
    #break


#%%
    
#%% making nice images
#import time
cmap = matplotlib.cm.get_cmap('jet')
ground_truth_basedir = '/home/rozmar/Data/Voltage_imaging/ground_truth' # location of the data
overwrite_images=True
subject_ids = os.listdir(ground_truth_basedir)
for subject_id in subject_ids:
    subject_dir = os.path.join(ground_truth_basedir,subject_id)
    cell_ids = os.listdir(subject_dir)
    for cell_id in cell_ids:
        cell_dir = os.path.join(subject_dir,cell_id)
        movies = sorted(os.listdir(cell_dir))
        for movie in movies:
# =============================================================================
#             try:
# =============================================================================
            #%
            movie_dir = os.path.join(ground_truth_basedir , subject_id , cell_id , movie)        
            
            with open(os.path.join(movie_dir,'movie_metadata.json')) as json_file:
                movie_metadata = json.load(json_file)
            frame_times = np.load(os.path.join(movie_dir,'frame_times.npy'))
            key_roi = {'subject_id': movie_metadata['subject_id'],
                       'session': movie_metadata['session'],
                       'movie_number': movie_metadata['movie_number'],
                       'motion_correction_method': 'VolPy',
                       'roi_type': 'VolPy'}
            #%
            dff,gt_roi_num = (imaging.ROI()*imaging_gt.GroundTruthROI()&key_roi).fetch1('roi_dff','roi_number')
            dff_all,roi_number_all = (imaging.ROI()&key_roi).fetch('roi_dff','roi_number')
            dff_all =dff_all[roi_number_all!=gt_roi_num]
            dff_list = [dff]
            dff_list.extend(dff_all)
            ephys_files_dir = os.path.join(movie_dir,'ephys')
            ephys_files = sorted(os.listdir(ephys_files_dir))
            sweep_time = list()
            sweep_response = list()
            sweep_stimulus = list()
            sweep_metadata = list()
            for ephys_file in ephys_files:
                if ephys_file[-3:]=='npz':
                    data_dict = np.load(os.path.join(ephys_files_dir,ephys_file))
                    sweep_time.append(data_dict['time'])
                    sweep_response.append(data_dict['voltage'])
                    sweep_stimulus.append(data_dict['stimulus'])
                    with open(os.path.join(ephys_files_dir,ephys_file[:-3]+'json')) as json_file:
                        sweep_metadata.append(json.load(json_file)) 
                        
            fig_dir = os.path.join(movie_dir,'figures')
            Path(fig_dir).mkdir(parents=True, exist_ok=True)  
            
            if len(os.listdir(fig_dir))>0 or overwrite_images:
                #%
                for file_now in os.listdir(fig_dir):
                    os.remove(os.path.join(fig_dir,file_now))
                
                #%
                fig=plt.figure()
                ax_ephys = fig.add_axes([0,0,2,.8])
                ax_stim = fig.add_axes([0,-.3,2,.2])
                ax_ap1 = fig.add_axes([0,-.8,2,.4])
                ax_ap2 = ax_ap1.twinx()
                ax_snr = fig.add_axes([0,-1.3,2,.4])
                for t,response,stimulus,metadata_now in zip(sweep_time,sweep_response,sweep_stimulus,sweep_metadata):
                    ax_ephys.plot(t,response,'k-')
                    ax_stim.plot(t,stimulus,'k-')
                    #%
                    key_cell ={'subject_id': metadata_now['subject_id'],
                               'session': metadata_now['session'],
                               'cell_number':metadata_now['cell_number'],
                               'sweep_number':metadata_now['sweep_number']}
                    ap_max_time, ap_amplitude,ap_halfwidth,ap_threshold,snratio = (imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()*ephysanal.ActionPotential()*ephysanal.ActionPotentialDetails()&key_cell&'ap_real=1'&'roi_type="VolPy"').fetch('ap_max_time','ap_amplitude','ap_halfwidth','ap_threshold','apwave_snratio')
                    ap_max_time = np.asarray(ap_max_time,float)
                    ax_ap2.plot(ap_max_time,ap_threshold-junction_potential,'ro')
                    ax_ap1.plot(ap_max_time,ap_amplitude,'ko')
                    ax_snr.plot(ap_max_time,snratio,'go')
                    
                    
                    #%
                if dff is not None:
                    ax_ophys = fig.add_axes([0,1,2,.8])
                    prevminval = 0
                    for dff_now,alpha_now in zip(dff_list,np.arange(1,1/(len(dff_list)+1),-1/(len(dff_list)+1))):
                        dfftoplotnow = dff_now + prevminval
                        ax_ophys.plot(frame_times,dfftoplotnow,'g-',alpha=alpha_now)
                        prevminval = np.min(dfftoplotnow) -.005
                    #ax_ophys.plot(frame_times,dff,'g-')
                    ax_ophys.autoscale(tight = True)
                    ax_ophys.invert_yaxis()
                    ax_ophys.set_xlim([frame_times[0],frame_times[-1]])
                    ax_ophys.set_ylabel('dF/F')
                    vals = ax_ophys.get_yticks()
                    ax_ophys.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
                
                ax_snr.set_xlim([frame_times[0],frame_times[-1]])
                ax_snr.set_ylim([0,ax_snr.get_ylim()[1]])
                ax_snr.set_ylabel('AP SNR')
                ax_ap1.set_xlim([frame_times[0],frame_times[-1]])
                ax_ap1.set_ylabel('AP amplitude (mV)')
                ax_ap2.set_ylabel('AP threshold (mV)')
                ax_ap2.yaxis.label.set_color('red')
                ax_ap2.tick_params(axis='y', colors='red')
    
                #ax_ap2.set_xlim([frame_times[0],frame_times[-1]])
                ax_ephys.autoscale(tight = True)
                ax_ephys.set_xlim([frame_times[0],frame_times[-1]])
                ax_ephys.set_ylabel('Membrane potential (mV)')
    
                ax_stim.set_xlim([frame_times[0],frame_times[-1]])
                ax_snr.set_xlabel('Time from obtaining whole cell (s)')
                ax_stim.set_ylabel('Stimulus (pA)')
                plt.show()
                #%
                if sweep_metadata[0]['neutralizationenable'] == 0:
                    fig.savefig(os.path.join(fig_dir,'whole_movie_NO_NEUTRALIZATION.png'), bbox_inches='tight')
                else:
                    fig.savefig(os.path.join(fig_dir,'whole_movie.png'), bbox_inches='tight')
                for t,response,stimulus,metadata_now in zip(sweep_time,sweep_response,sweep_stimulus,sweep_metadata):
                    ax_ephys.autoscale(tight = True)
                    ax_ophys.autoscale(tight = True)
                    ax_stim.set_xlim([t[0],t[-1]])
                    ax_ephys.set_xlim([t[0],t[-1]])
                    ax_ophys.set_xlim([t[0],t[-1]])
                    ax_snr.set_xlim([t[0],t[-1]])
                    ax_ap1.set_xlim([t[0],t[-1]])
                    ax_ophys.set_title('')
                    fig.savefig(os.path.join(fig_dir,'sweep_{}.png'.format(metadata_now['sweep_number'])), bbox_inches='tight')
                    stimidx = np.argmin(np.diff(stimulus))
                    stimlimits= [t[stimidx]-.05, t[stimidx]+.15]
                    ax_stim.set_xlim(stimlimits)
                    ax_ephys.set_xlim(stimlimits)
                    ax_ophys.set_xlim(stimlimits)
                    ax_snr.set_xlim(stimlimits)
                    ax_ap1.set_xlim(stimlimits)
                    #set ylimits
                    i = np.where( (t > stimlimits[0]) &  (t < stimlimits[1]) )[0]
                    ax_ephys.set_ylim( response[i].min(), response[i].max())
                    i = np.where( (frame_times > stimlimits[0]) &  (frame_times < stimlimits[1]) )[0]
                    try:
                        ax_ophys.set_ylim( dff[i].min(), dff[i].max())
                    except:
                        pass
                    ax_ophys.invert_yaxis()
                    key_cell ={'subject_id': metadata_now['subject_id'],
                               'session': metadata_now['session'],
                               'cell_number':metadata_now['cell_number'],
                               'sweep_number':metadata_now['sweep_number']}
                    rs_tot,rs_bridged = (ephysanal.SweepSeriesResistance()&key_cell).fetch1('series_resistance','series_resistance_residual')
                    if rs_tot == None:
                        ax_ophys.set_title('RS: {}MOhm - bridged: {}MOhm'.format(0,0))
                    else:
                        ax_ophys.set_title('RS: {}MOhm - bridged: {}MOhm'.format(int(rs_tot),int(rs_bridged)))
                    fig.savefig(os.path.join(fig_dir,'sweep_{}_squarepulse.png'.format(metadata_now['sweep_number'])), bbox_inches='tight')
                    #%
                roimasks_all = list()
                roimasks = (imaging.ROI()&key_roi).fetch('roi_mask')
                for roimask_now in roimasks:
                    roimasks_all.append(roimask_now/np.max(roimask_now))
                roimasks = np.sum(np.asarray(roimasks_all),0)
                meanimage = (imaging.RegisteredMovie()&key_roi).fetch1('registered_movie_mean_image')
                  #%%
                fig_images = plt.figure()
                ax_meanimage = fig_images.add_axes([0,0,1,1])
                im = ax_meanimage.imshow(meanimage,cmap=cmap)
                divider = make_axes_locatable(ax_meanimage)
                cax = divider.append_axes("bottom", size="20%", pad=0.5)
                plt.colorbar(im, cax=cax, orientation = "horizontal")
                
                ax_meanimage.set_title('mean image')
                ax_ROIs = fig_images.add_axes([1.1,0.08,1,1])
                ax_ROIs.imshow(roimasks,cmap=cmap)
                ax_ROIs.set_title('ROIs')
                #fig_images.colorbar(im)
                fig_images.savefig(os.path.join(fig_dir,'movie_mean_image.png'), bbox_inches='tight')
                #%%
# =============================================================================
#             except:
#                 print('no ROI FOUND')#fig_2=plt.figure()
#                 print(key_roi)
#                 time.sleep(10000)
# =============================================================================
            
                # TODO plot RS, ap amplitude, halfwidth, SN ratio
             #%%
            
            #time.sleep(1000)
        

            
            
            