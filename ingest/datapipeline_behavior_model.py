from datetime import datetime, timedelta, time
import pandas as pd
from pathlib import Path
from pybpodgui_api.models.project import Project
import numpy as np
import Behavior.behavior_rozmar as behavior_rozmar
from Behavior.foraging_model import foraging_model
import time as timer
import decimal
#% connect to server
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment,ephys_patch
from pipeline import behavioranal, ephysanal
import ray
#%%

@ray.remote
def model_and_populate_a_session(subject_now,subject_id,session):
    print('session '+str(session))
    p_reward_L, p_reward_R, n_trials = foraging_model.generate_block_structure(n_trials_base=80,n_trials_sd=10,blocknum = 8, reward_ratio_pairs=np.array([[.4,.05],[.3857,.0643],[.3375,.1125],[.225,.225]]))
    if subject_now == 'leaky3t5it30h':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'clever',
                                                  min_rewardnum = 30, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 00.00, 
                                                  softmax_temperature =5,
                                                  plot = False)
    elif subject_now == 'leaky3t3it30h':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'clever',
                                                  min_rewardnum = 30, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 0, 
                                                  softmax_temperature =3,
                                                  plot = False)
    elif subject_now == 'W-St-L-Sw':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'win_stay-loose_switch',
                                                  min_rewardnum = 3, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 00.01, 
                                                  plot = False)
    elif subject_now == 'W-St-L-Rnd':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'win_stay-loose_random',
                                                  min_rewardnum = 3, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 00.01, 
                                                  filter_constant = .05,
                                                  plot = False)
    elif subject_now == 'leaky3t.05c15h':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'clever',
                                                  min_rewardnum = 15, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 00.0, 
                                                  filter_constant = .05,
                                                  plot = False)
    elif subject_now == 'leaky3t.05c5h':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'clever',
                                                  min_rewardnum = 5, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 00.0, 
                                                  filter_constant = .05,
                                                  plot = False)
    elif subject_now == 'leaky3t.05c30h':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'clever',
                                                  min_rewardnum = 30, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 00.0, 
                                                  filter_constant = .05,
                                                  plot = False)
    elif subject_now == 'cheater':
        rewards, choices = foraging_model.run_task(p_reward_L,
                                                  p_reward_R,
                                                  n_trials,
                                                  unchosen_rewards_to_keep = 1,
                                                  subject = 'perfect',
                                                  min_rewardnum = 30, 
                                                  filter_tau_fast = 3,
                                                  filter_tau_slow = 100, 
                                                  filter_tau_slow_amplitude = 00.0, 
                                                  filter_constant = .05,
                                                  plot = False)
        
    else:
        print('unknown model')
    sessiondata = {
            'subject_id': subject_id,
            'session' : session,
            'session_date' : datetime.now().strftime('%Y-%m-%d'),
            'session_time' : datetime.now().strftime('%H:%M:%S'),
            'username' : experimenter,
            'rig': setupname
            }
    experiment.Session().insert1(sessiondata)
    trialssofar = 0
    columns = ['subject_id','session','block','block_uid','block_start_time','p_reward_left','p_reward_right']
    df_sessionblockdata = pd.DataFrame(data = np.zeros((len(p_reward_L),len(columns))),columns = columns)
    for blocknum,(p_L,p_R,trialnum) in enumerate(zip(p_reward_L,p_reward_R,n_trials),1):
        df_sessionblockdata.loc[blocknum-1,'subject_id'] = subject_id 
        df_sessionblockdata.loc[blocknum-1,'session'] = session 
        df_sessionblockdata.loc[blocknum-1,'block'] = blocknum 
        df_sessionblockdata.loc[blocknum-1,'block_uid'] = blocknum 
        df_sessionblockdata.loc[blocknum-1,'block_start_time'] = trialssofar 
        df_sessionblockdata.loc[blocknum-1,'p_reward_left'] = p_L 
        df_sessionblockdata.loc[blocknum-1,'p_reward_right'] = p_R 
        trialssofar += trialnum
    experiment.SessionBlock().insert(df_sessionblockdata.to_records(index=False), allow_direct_insert=True)
    columns_sessiontrial = ['subject_id','session','trial','trial_uid','trial_start_time','trial_stop_time']
    df_sessiontrialdata = pd.DataFrame(data = np.zeros((len(rewards),len(columns_sessiontrial))),columns = columns_sessiontrial)
    columns_behaviortrial = ['subject_id','session','trial','task','task_protocol','trial_choice','early_lick','outcome','block']
    df_behaviortrialdata = pd.DataFrame(data = np.zeros((len(rewards),len(columns_behaviortrial))),columns = columns_behaviortrial)
    for trialnum,(reward,choice) in enumerate(zip(rewards,choices),1):
        df_sessiontrialdata.loc[trialnum-1,'subject_id'] = subject_id 
        df_sessiontrialdata.loc[trialnum-1,'session'] = session 
        df_sessiontrialdata.loc[trialnum-1,'trial'] = trialnum 
        df_sessiontrialdata.loc[trialnum-1,'trial_uid'] = trialnum 
        df_sessiontrialdata.loc[trialnum-1,'trial_start_time'] = trialnum - .9 
        df_sessiontrialdata.loc[trialnum-1,'trial_stop_time'] = trialnum - .1 
        

        #% outcome
        if reward:
            outcome = 'hit'
        else:
            outcome = 'miss'
        if choice == 1:
            trial_choice = 'right'
        else:
            trial_choice = 'left'
        task = 'foraging'
        task_protocol = 10
        df_behaviortrialdata.loc[trialnum-1,'subject_id'] = subject_id 
        df_behaviortrialdata.loc[trialnum-1,'session'] = session 
        df_behaviortrialdata.loc[trialnum-1,'trial'] = trialnum 
        df_behaviortrialdata.loc[trialnum-1,'task'] = task 
        df_behaviortrialdata.loc[trialnum-1,'task_protocol'] = task_protocol 
        df_behaviortrialdata.loc[trialnum-1,'trial_choice'] = trial_choice 
        df_behaviortrialdata.loc[trialnum-1,'early_lick'] = 'no early' 
        df_behaviortrialdata.loc[trialnum-1,'outcome'] = outcome 
        df_behaviortrialdata.loc[trialnum-1,'block'] = np.argmax(np.cumsum(n_trials)>=trialnum)+1 
    experiment.SessionTrial().insert(df_sessiontrialdata.to_records(index=False), allow_direct_insert=True)
    experiment.BehaviorTrial().insert(df_behaviortrialdata.to_records(index=False), allow_direct_insert=True)


sessionnumber = 30
for subject_now  in ['leaky3t.05c30h','leaky3t.05c15h','leaky3t.05c5h','W-St-L-Rnd','W-St-L-Sw','leaky3t5it30h','leaky3t3it30h','cheater']:
    #subject_now = 'leaky3ms.05c30t'#'leaky3ms.05c15t'#'leaky3ms.05c30t'#'W-St-L-Rnd'#'W-St-L-Sw' #'leakyint_9ms+c'#'leakyint_3ms+c'#'
    setupname = 'virtual_setup'
    experimenter  = 'rozsam'
    print('subject: ',subject_now)
    
    if len(lab.WaterRestriction()&'water_restriction_number = "'+subject_now+'"')>0:
        subject_id_to_del = (lab.WaterRestriction()&'water_restriction_number = "'+subject_now+'"').fetch('subject_id')[0]
        dj.config['safemode'] = False
        (lab.Subject() & 'subject_id = '+str(subject_id_to_del)).delete()
        dj.config['safemode'] = True
    for subject_id in range(100):
        if len(lab.Subject&'subject_id = '+str(subject_id)) == 0:
            break
    #%
    subjectdata = {
            'subject_id': subject_id,
            'cage_number': 0,
            'date_of_birth': datetime.now().strftime('%Y-%m-%d'),
            'sex': 'Unknown',
            'username': experimenter,
            }
    lab.Subject().insert1(subjectdata)
    wrdata = {
            'subject_id':subject_id,
            'water_restriction_number': subject_now,
            'cage_number': 0,
            'wr_start_date': datetime.now().strftime('%Y-%m-%d'),
            'wr_start_weight': 0,
            }
    lab.WaterRestriction().insert1(wrdata)
    
    ray.init(num_cpus = 6)
    result_ids = []
    for session in range(1,sessionnumber):    
        
        result_ids.append(model_and_populate_a_session.remote(subject_now,subject_id,session))        
    ray.get(result_ids)            
    ray.shutdown()        
        # do the modeling
    #%%
    