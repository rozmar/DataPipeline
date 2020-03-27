import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
import pipeline.ephys_patch as ephys_patch
import pipeline.ephysanal as ephysanal
import pipeline.imaging as imaging
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging_gt'),locals()) # TODO ez el van baszva

import numpy as np
import scipy
#%%
@schema
class CellMovieCorrespondance(dj.Computed):
    definition = """
    -> ephys_patch.Cell
    -> imaging.Movie
    ---
    sweep_numbers = null                   : longblob # sweep numbers that 
    """
    def make(self, key):
        #%
       # key = { 'subject_id': 462149, 'session':1,'cell_number':1,'movie_number':11}
        session_time = (experiment.Session()&key).fetch('session_time')[0]
        cell_time = (ephys_patch.Cell()&key).fetch('cell_recording_start')[0]
        cell_sweep_start_times =  (ephys_patch.Sweep()&key).fetch('sweep_start_time')
        cell_sweep_end_times =  (ephys_patch.Sweep()&key).fetch('sweep_end_time')
        time_start = float(np.min(cell_sweep_start_times))+ cell_time.total_seconds() - session_time.total_seconds()
        time_end = float(np.max(cell_sweep_end_times))+ cell_time.total_seconds() - session_time.total_seconds()
        try:
            movie = (imaging.Movie())&key & 'movie_start_time > '+str(time_start) & 'movie_start_time < '+str(time_end)
            sweep_start_times,sweep_end_times,sweep_nums = (ephys_patch.Sweep()&key).fetch('sweep_start_time','sweep_end_time','sweep_number')
            sweep_start_times = np.asarray(sweep_start_times,float)+ cell_time.total_seconds() - session_time.total_seconds()
            sweep_end_times = np.asarray(sweep_end_times,float)+ cell_time.total_seconds() - session_time.total_seconds()
            #for movie in movies_now:
            frametimes = (imaging.MovieFrameTimes&movie).fetch1('frame_times')
            needed_start_time = frametimes[0]
            needed_end_time = frametimes[-1]
            sweep_nums_needed = sweep_nums[((sweep_start_times > needed_start_time) & (sweep_start_times < needed_end_time)) |
                                    ((sweep_end_times > needed_start_time) & (sweep_end_times < needed_end_time)) | 
                                    ((sweep_end_times > needed_end_time) & (sweep_start_times < needed_start_time)) ]
            if len(sweep_nums_needed)>0:
                key['sweep_numbers'] = sweep_nums_needed
                self.insert1(key,skip_duplicates=True)
        except:
            pass
        
            
        #%
@schema
class ROIEphysCorrelation(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> imaging.ROI
    -> ephys_patch.Sweep
    ---
    time_lag                        : float #ms   
    corr_coeff                      : float #-1 - 1
    """

@schema
class ROIAPWave(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """ # this is the optical AP waveform relative to the real AP peak
    -> imaging.ROI
    -> ephysanal.ActionPotential
    ---
    apwave_time                     : longblob
    apwave_dff                      : longblob
    apwave_snratio                  : float
    apwave_peak_amplitude           : float
    apwave_noise                    : float
    apwave_f0                       : float
    """

@schema
class GroundTruthROI(dj.Computed):
    definition = """ # this is the optical AP waveform relative to the real AP peak
    -> ephys_patch.Cell
    -> imaging.ROI
    --- 
    ephys_matched_ap_times                  : longblob # in seconds, from the start of the session
    ophys_matched_ap_times                  : longblob # in seconds, from the start of the session
    ephys_unmatched_ap_times                : longblob # in seconds, from the start of the session
    ophys_unmatched_ap_times                : longblob # in seconds, from the start of the session
    """
    def make(self, key):
        #%
        #key = {'subject_id': 454597, 'session': 1, 'cell_number': 0, 'motion_correction_method': 'Matlab', 'roi_type': 'SpikePursuit', 'roi_number': 1}
        #key = {'subject_id': 456462, 'session': 1, 'cell_number': 5, 'movie_number': 3, 'motion_correction_method': 'VolPy', 'roi_type': 'VolPy', 'roi_number': 1}
        if len(ROIEphysCorrelation&key)>0:#  and key['roi_type'] == 'SpikePursuit' #only spikepursuit for now..
            key_to_compare = key.copy()
            del key_to_compare['roi_number']
            #print(key)
            #%
            roinums = np.unique((ROIEphysCorrelation()&key_to_compare).fetch('roi_number'))
            snratios_mean = list()
            snratios_median = list()
            snratios_first50 = list()
            for roinum_now in roinums:
                
                snratios = (ROIAPWave()&key_to_compare &'roi_number = {}'.format(roinum_now)).fetch('apwave_snratio')
                snratios_mean.append(np.mean(snratios))
                snratios_median.append(np.median(snratios))
                snratios_first50.append(np.mean(snratios[:50]))
                
            
            #%%
            if np.max((ROIEphysCorrelation()&key).fetch('roi_number')) == roinums[np.argmax(snratios_first50)]:#np.max((imaging_gt.ROIEphysCorrelation()&key).fetch('roi_number')) == np.min((imaging_gt.ROIEphysCorrelation&key_to_compare).fetch('roi_number')):#np.max(np.abs((ROIEphysCorrelation&key).fetch('corr_coeff'))) == np.max(np.abs((ROIEphysCorrelation&key_to_compare).fetch('corr_coeff'))):
                print('this is it')
                print(key['roi_type'])
                cellstarttime = (ephys_patch.Cell()&key).fetch1('cell_recording_start')
                sessionstarttime = (experiment.Session()&key).fetch1('session_time')
                aptimes = np.asarray((ephysanal.ActionPotential()&key).fetch('ap_max_time'),float)+(cellstarttime-sessionstarttime).total_seconds()
                sweep_start_times,sweep_end_times = (ephys_patch.Sweep()&key).fetch('sweep_start_time','sweep_end_time')
                sweep_start_times = np.asarray(sweep_start_times,float)+(cellstarttime-sessionstarttime).total_seconds()
                sweep_end_times = np.asarray(sweep_end_times,float)+(cellstarttime-sessionstarttime).total_seconds()
                frame_timess,roi_spike_indicess=(imaging.MovieFrameTimes()*imaging.Movie()*imaging.ROI()&key).fetch('frame_times','roi_spike_indices')
                movie_start_times=list()
                movie_end_times = list()
                roi_ap_times = list()
                for frame_times,roi_spike_indices in zip(frame_timess,roi_spike_indicess):
                    movie_start_times.append(frame_times[0])
                    movie_end_times.append(frame_times[-1])
                    roi_ap_times.append(frame_times[roi_spike_indices])
                movie_start_times = np.sort(movie_start_times)
                movie_end_times = np.sort(movie_end_times)
                roi_ap_times=np.sort(np.concatenate(roi_ap_times))
                #%
                ##delete spikes in optical traces where there was no ephys recording
                for start_t,end_t in zip(np.concatenate([sweep_start_times,[np.inf]]),np.concatenate([[0],sweep_end_times])):
                    idxtodel = np.where((roi_ap_times>end_t) & (roi_ap_times<start_t))[0]
                    if len(idxtodel)>0:
                        roi_ap_times = np.delete(roi_ap_times,idxtodel)
                ##delete spikes in ephys traces where there was no imaging
                for start_t,end_t in zip(np.concatenate([movie_start_times,[np.inf]]),np.concatenate([[0],movie_end_times])):
                    idxtodel = np.where((aptimes>end_t) & (aptimes<start_t))[0]
                    if len(idxtodel)>0:
                        #print(idxtodel)
                        aptimes = np.delete(aptimes,idxtodel)
                        #%
                D = np.zeros([len(aptimes),len(roi_ap_times)])
                for idx,apt in enumerate(aptimes):
                    D[idx,:]=(roi_ap_times-apt)*1000
                D_test = np.abs(D)    
                D_test[D_test>15]=1000
                D_test[D<-1]=1000
                X = scipy.optimize.linear_sum_assignment(D_test)    
                #%
                cost = D_test[X[0],X[1]]
                unmatched = np.where(cost == 1000)[0]
                X0_final = np.delete(X[0],unmatched)
                X1_final = np.delete(X[1],unmatched)
                ephys_ap_times = aptimes[X0_final]
                ophys_ap_times = roi_ap_times[X1_final]
                false_positive_time_imaging = list()
                for roi_ap_time in roi_ap_times:
                    if roi_ap_time not in ophys_ap_times:
                        false_positive_time_imaging.append(roi_ap_time)
                false_negative_time_ephys = list()
                for aptime in aptimes:
                    if aptime not in ephys_ap_times:
                        false_negative_time_ephys.append(aptime)
                
                key['ephys_matched_ap_times'] = ephys_ap_times
                key['ophys_matched_ap_times'] = ophys_ap_times
                key['ephys_unmatched_ap_times'] = false_negative_time_ephys
                key['ophys_unmatched_ap_times'] = false_positive_time_imaging
                #print(imaging.ROI()&key)
                #print([len(aptimes),'vs',len(roi_ap_times)])
                #%%
                self.insert1(key,skip_duplicates=True)

            #else:
                
                #print('this is not it')
            
        
    
    