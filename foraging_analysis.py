from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import datajoint as dj
from pipeline import pipeline_tools, lab, experiment, behavioranal
dj.conn()
import matplotlib.pyplot as plt
import decimal

#%% foraging tuning curves based on local reward rate!!! IMPLEMENT ME!!!
local_filter = np.ones(10)
wr_name = 'FOR08'
session = 4
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
key = {
       'subject_id':subject_id,
       'session': session,
       }
df_choices = pd.DataFrame(experiment.BehaviorTrial()&key)
#%
reward_ratio_binnum = 10
filter_now = local_filter[::-1]




right_choice = (df_choices['trial_choice'] == 'right').values
left_choice = (df_choices['trial_choice'] == 'left').values
right_reward = ((df_choices['trial_choice'] == 'right')&(df_choices['outcome'] == 'hit')).values
left_reward = ((df_choices['trial_choice'] == 'left')&(df_choices['outcome'] == 'hit')).values

right_choice_conv = np.convolve(right_choice , filter_now,mode = 'valid')
left_choice_conv = np.convolve(left_choice , filter_now,mode = 'valid')
right_reward_conv = np.convolve(right_reward , filter_now,mode = 'valid')
left_reward_conv = np.convolve(left_reward , filter_now,mode = 'valid')

right_choice = right_choice[len(filter_now)-1:]
left_choice = left_choice[len(filter_now)-1:]

choice_num = np.ones(len(left_choice))
choice_num[:]=np.nan
choice_num[left_choice] = 0
choice_num[right_choice] = 1

reward_ratio_right = right_reward_conv/right_choice_conv
reward_ratio_right[np.isnan(reward_ratio_right)] = 0
reward_ratio_left = left_reward_conv/left_choice_conv
reward_ratio_left[np.isnan(reward_ratio_left)] = 0
reward_ratio_combined = reward_ratio_right/(reward_ratio_right+reward_ratio_left)


todel = np.isnan(reward_ratio_combined)
reward_ratio_combined = reward_ratio_combined[~todel]
choice_num = choice_num[~todel]
todel = np.isnan(choice_num)
reward_ratio_combined = reward_ratio_combined[~todel]
choice_num = choice_num[~todel]

bottoms = np.arange(0,100, 100/reward_ratio_binnum)
tops = np.arange(100/reward_ratio_binnum,100.005, 100/reward_ratio_binnum)

reward_ratio_mean = list()
reward_ratio_sd = list()
choice_ratio_mean = list()
choice_ratio_sd = list()
for bottom,top in zip(bottoms,tops):
    minval = np.percentile(reward_ratio_combined,bottom)
    maxval = np.percentile(reward_ratio_combined,top)
    if minval == maxval:
        idx = (reward_ratio_combined== minval)
    else:
        idx = (reward_ratio_combined>= minval) & (reward_ratio_combined < maxval)
    reward_ratio_mean.append(np.mean(reward_ratio_combined[idx]))
    reward_ratio_sd.append(np.std(reward_ratio_combined[idx]))
    choice_ratio_mean.append(np.mean(choice_num[idx]))
    choice_ratio_sd.append(np.std(choice_num[idx]))
    
plt.errorbar(reward_ratio_mean,choice_ratio_mean,choice_ratio_sd,reward_ratio_sd)    



#%% foraging tuning curves based on block
wr_name = 'FOR01'
minsession = 8
mintrialnum = 50
metricnames = ['block_choice_ratio','block_choice_ratio_first_tertile','block_choice_ratio_second_tertile','block_choice_ratio_third_tertile']
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
key = {
       'subject_id':subject_id,
       #'session': session
       }

pd_choice_reward_rate = pd.DataFrame((experiment.SessionBlock()*behavioranal.BlockRewardRatio()*behavioranal.BlockChoiceRatio()*behavioranal.BlockAutoWaterCount()) & key)
#%
pd_choice_reward_rate['block_relative_value']=pd_choice_reward_rate['p_reward_right']/(pd_choice_reward_rate['p_reward_right']+pd_choice_reward_rate['p_reward_left'])
pd_choice_reward_rate['total_reward_rage']=(pd_choice_reward_rate['p_reward_right']+pd_choice_reward_rate['p_reward_left'])
needed = (pd_choice_reward_rate['total_reward_rage']< .5) & (pd_choice_reward_rate['session']>= minsession) & (pd_choice_reward_rate['block_choice_ratio']>-1) & (pd_choice_reward_rate['block_autowater_count']==0) & (pd_choice_reward_rate['block_length'] >= mintrialnum)
pd_choice_reward_rate = pd_choice_reward_rate[needed] # unwanted blocks are deleted
#%
fig=plt.figure()

ax_blocklenght=fig.add_axes([0,1,1,.8])
out = ax_blocklenght.hist(pd_choice_reward_rate['block_length'],30)
ax_blocklenght.set_xlabel('Block length (trials)')
ax_blocklenght.set_ylabel('Count')
ax_blocklenght.set_title(wr_name)
for idx,metricname in enumerate(metricnames):
    relvals = np.sort(pd_choice_reward_rate['block_relative_value'].unique())
    choice_ratio_mean = list()
    choice_ratio_sd = list()
    choice_ratio_median = list()
    reward_rate_value = list()
    for relval in relvals:
        choice_rate_vals = pd_choice_reward_rate[metricname][pd_choice_reward_rate['block_relative_value']==relval]
        choice_ratio_mean.append(choice_rate_vals.mean())
        choice_ratio_median.append(choice_rate_vals.median())
        choice_ratio_sd.append(float(np.std(choice_rate_vals.to_numpy())))
        reward_rate_value.append(float(relval))
    #%
    
    ax_1=fig.add_axes([0,-idx,1,.8])
    ax_1.errorbar(reward_rate_value,choice_ratio_mean,choice_ratio_sd,color = 'black',linewidth = 3,marker='o',ms=9)
    ax_1.plot(pd_choice_reward_rate['block_relative_value'],pd_choice_reward_rate[metricname],'o',markersize = 3,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
    ax_1.plot([0,1],[0,1],'k-')
    ax_1.set_ylim([0, 1])
    ax_1.set_xlim([0, 1])
    ax_1.set_xlabel('relative value (p_R/(p_R+p_L))')
    ax_1.set_ylabel('relative choice (c_R/(c_R+c_L))')
    ax_1.set_title(metricname)
    ax_2=fig.add_axes([1.2,-idx,1,.8])
    ax_2.plot(pd_choice_reward_rate['block_length'],pd_choice_reward_rate[metricname]-pd_choice_reward_rate['block_relative_value'],'ko',markersize = 3)
#%% show weight and water consumption of subjects over time
df_subject_wr=pd.DataFrame(lab.WaterRestriction() * experiment.Session() * experiment.SessionDetails)
subject_names = df_subject_wr['water_restriction_number'].unique()
subject_names.sort()
fig=plt.figure()
ax_weight=fig.add_axes([0,0,1,.8])
for i,subject in enumerate(subject_names):
    idx = df_subject_wr['water_restriction_number'] == subject
    weights = df_subject_wr['session_weight'][idx] /df_subject_wr['wr_start_weight'][idx]
    ax_weight.plot(range(1,len(weights)+1),weights.values)
vals = ax_weight.get_yticks()
ax_weight.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
ax_weight.set_ylabel('Percentage of baseline weight')
ax_weight.set_xlabel('Session number')
ax_weight.legend(subject_names)
ax_weight.set_title('Weight change over training')

ax_water = fig.add_axes([0,-1,1,.8])
for i,subject in enumerate(subject_names):
    idx = df_subject_wr['water_restriction_number'] == subject
    water = df_subject_wr['session_water_earned'][idx]
    ax_water.plot(range(1,len(water)+1),water.values,linewidth=len(subject_names)+1-i)
ax_water.set_ylabel('Consumed water (ml)')
ax_water.set_xlabel('Session number')
ax_water.set_ylim(0, .6)
ax_water.legend(subject_names)
ax_water.set_title('Water consumption over training')
# show task type over time
# =============================================================================
# df_taskprotocol = pd.DataFrame(behavioranal.SessionTrainingType())
# df_taskprotocol_types = pd.DataFrame(experiment.TaskProtocol())
# 
# ax_protocol = fig.add_axes([0,-2,1,.8])
# for i,subject in enumerate(subject_names):
#     idx = df_subject_wr['water_restriction_number'] == subject
#     protocolnums = df_taskprotocol['session_task_protocol'][idx]
#     ax_protocol.plot(range(1,len(protocolnums)+1),protocolnums.values,linewidth=len(subject_names)+1-i)
# =============================================================================

#%
ax_trialnum=fig.add_axes([1.2,0,1,.8])
ax_earlylick=fig.add_axes([1.2,-1,1,.8])
for wr_name in subject_names:
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {'subject_id':subject_id}
    sessionnum = len(experiment.Session()&key)
    trialnums = list()
    earlylickrates = list()
    for session in range(1,sessionnum+1):    
        key = {'session':session,'subject_id':subject_id}    
        trialnum = len(experiment.SessionTrial() & key)
        trialnums.append(trialnum)    
        df_earlylick = pd.DataFrame((experiment.BehaviorTrial()&key).fetch('early_lick'))
        early_lick_rate = (df_earlylick == 'early').values.sum()/trialnum
        earlylickrates.append(early_lick_rate)
    ax_trialnum.plot(range(1,len(trialnums)+1),trialnums,linewidth=len(subject_names)+1-i)
    ax_earlylick.plot(range(1,len(earlylickrates)+1),earlylickrates,linewidth=len(subject_names)+1-i)
ax_trialnum.legend(subject_names)
ax_trialnum.set_xlabel('Session number')
ax_trialnum.set_ylabel('Number of trials')
ax_trialnum.set_title('Number of trials during each session')

ax_earlylick.legend(subject_names)
ax_earlylick.set_xlabel('Session number')
ax_earlylick.set_ylabel('Early lick rate')
ax_earlylick.set_title('Early lick rate during each session')


#%%
fig=plt.figure()
ax_autowater=fig.add_axes([0,0,1,.8])
for wr_name in subject_names:
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {'subject_id':subject_id}
    sessionnum = len(experiment.Session()&key)
    autowater_nums = list()
    
    for session in range(1,sessionnum+1):  
        key = {'session':session,'subject_id':subject_id}    
        autowater_num = len((experiment.TrialNote()&key))
        autowater_nums.append(autowater_num)
    ax_autowater.plot(range(1,len(autowater_nums)+1),autowater_nums,linewidth=len(subject_names)+1-i)    
ax_trialnum.legend(subject_names)
ax_autowater.set_xlabel('Session number')
ax_autowater.set_ylabel('Number of trials with autowater on')
ax_autowater.set_title('Autowater')   
#%% reaction times for the last session:
session = experiment.Session().fetch('session').max()
key = {'session':session}
df_reactiontime = pd.DataFrame((behavioranal.SessionReactionTimeHistogram() & key)*(lab.WaterRestriction &key))
fig=plt.figure()
ax_RT = list()
for i,subject in enumerate(subject_names):
    ax_RT.append(fig.add_axes([0,-i,1,.8]))
    idx = df_reactiontime['water_restriction_number'] == subject
    bins = df_reactiontime['reaction_time_bins'][idx].values[0][1:]
    edges = [np.min(bins),np.max(bins)]
    bins = bins - (bins[1] -bins[0])/2
    vals = df_reactiontime['reaction_time_values_all_trials'][idx].values.tolist()[0]
    ax_RT[-1].bar(bins,vals, (bins[1] -bins[0])/2)
    ax_RT[-1].set_xlim(edges)
    ax_RT[-1].set_title(subject)
    ax_RT[-1].set_xlabel('Reaction time (s)')
    ax_RT[-1].set_ylabel('Trial count')

#%% licks on miss trials
df_misslicknums = pd.DataFrame()
fig=plt.figure()
ax_misslick = fig.add_axes([0,0,1,.8])
for wr_name in subject_names:
    #%
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {'subject_id':subject_id}   
    df_lickrhythm = pd.DataFrame((behavioranal.SessionLickRhythmHistogram() & key)*(lab.WaterRestriction &key))
    #%
    
    sessionnum = list()
    licknum = list()
    for i,session in enumerate(df_lickrhythm['session']):
        idx = df_lickrhythm['session'] == session
        bins = df_lickrhythm['lick_rhythm_bins'][idx].values[0][1:]
        edges = [np.min(bins),np.max(bins)]
        bins = bins - (bins[1] -bins[0])/2 
        vals = df_lickrhythm['lick_rhythm_values_miss_trials'][idx].values.tolist()[0]
        sessionnum.append(session)
        licknum.append(vals[bins>.015].sum())       
    ax_misslick.plot(sessionnum,licknum)
ax_misslick.set_ylabel('Summed extra licks on miss trials')
ax_misslick.set_xlabel('Session number')
ax_misslick.legend(subject_names)
ax_misslick.set_title('Lick count on miss trials')

#%% block switches
def plot_block_switches(wr_name = None):
    if wr_name == None:
        df_block_starts = pd.DataFrame(behavioranal.SessionBlockSwitchChoices() & 'session > 5')
        wr_name = 'All subjects'
    else:
        subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]

        df_block_starts = pd.DataFrame(behavioranal.SessionBlockSwitchChoices() & 'session > 5' & 'subject_id = ' + str(subject_id))
        print(subject_id)
    contrast_edges = [.1,.25,.3]
    bigchoicematrix = None
    p_r_change = None
    p_l_change = None
    p_r_next = None
    p_r_prev = None
    next_block_length = None
    for idx, line in df_block_starts.iterrows():
        if len(line['p_r_change']) > 0:
            if bigchoicematrix is None:
                bigchoicematrix = line['choices_matrix']        
                p_r_change  = line['p_r_change']        
                p_l_change  = line['p_l_change']  
                next_block_length  = line['block_length_next']   
                p_r_next  = line['p_r_next']   
                p_r_prev  = line['p_r_prev']   
            else:
                bigchoicematrix = np.concatenate((bigchoicematrix,line['choices_matrix']))
                p_r_change  =  np.concatenate((p_r_change,line['p_r_change']))
                p_l_change  =  np.concatenate((p_l_change,line['p_l_change']))
                next_block_length  =  np.concatenate((next_block_length,line['block_length_next']))       
                p_r_next  =  np.concatenate((p_r_next,line['p_r_next']))       
                p_r_prev  =  np.concatenate((p_r_prev,line['p_r_prev']))   
    
    fig=plt.figure()
    ax1=fig.add_axes([0,0,.8,.8])
    ax1.hist([p_r_change,p_l_change],40)
    ylimedges = ax1.get_ylim()
    ax1.set_title('Contrast between blocks - Subject: '+wr_name)
    ax1.set_ylabel('Count of block switches')
    ax1.set_xlabel('Change in probability')
    ax1.legend(['right lickport','left lickport','contrast groups'])
    ax1.plot(contrast_edges,np.ones(len(contrast_edges))*np.mean(ylimedges),'k|',markersize = 500)
    ax1.set_xlim(0,.4)
    
    ax2=fig.add_axes([1,0,.8,.8])
    ax2.hist(next_block_length,40)
    ax2.set_title('Block length distribution - '+wr_name)
    ax2.set_ylabel('# of blocks')
    ax2.set_xlabel('# of trials in each block')
    
    ax3=fig.add_axes([0,-1,.8,.8])
    idx = (p_r_change >0) & (p_r_change >.3) & (np.abs(p_r_change) <1)
    ax3.plot(np.arange(-30,50),np.ones(80)*.5,'k-')
    ax3.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    idx = (p_l_change > 0) & (p_l_change > .3) & (np.abs(p_r_change) <1)
    ax3.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    ax3.set_title('High contrast - '+wr_name)
    ax3.set_xlabel('Trials relative to block switch')
    ax3.set_ylabel('Average choice')
    ax3.set_ylim(.1,.9)
    
    ax4=fig.add_axes([1,-1,.8,.8])
    idx = (p_r_change >.25) & (p_r_change <.3) & (np.abs(p_r_change) <1)
    ax4.plot(np.arange(-30,50),np.ones(80)*.5,'k-')
    ax4.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    idx = (p_l_change > .25) & (p_l_change < .3) & (np.abs(p_r_change) <1)
    ax4.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    ax4.set_title('Intermediate contrast - '+wr_name)
    ax4.set_xlabel('Trials relative to block switch')
    ax4.set_ylabel('Average choice')
    ax4.set_ylim(.1,.9)
    
    ax5=fig.add_axes([0,-2,.8,.8])
    idx = (p_r_change >.1) & (p_r_change <.25) & (np.abs(p_r_change) <1)
    ax5.plot(np.arange(-30,50),np.ones(80)*.5,'k-')
    ax5.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    idx = (p_l_change > .1) & (p_l_change < .25) & (np.abs(p_r_change) <1)
    ax5.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    ax5.set_title('Low contrast - '+wr_name)
    ax5.set_xlabel('Trials relative to block switch')
    ax5.set_ylabel('Average choice')
    ax5.set_ylim(.1,.9)
    
    ax6=fig.add_axes([1,-2,.8,.8])
    idx = (p_r_change >.1) & (p_r_change <1) & (np.abs(p_r_change) <1)
    ax6.plot(np.arange(-30,50),np.ones(80)*.5,'k-')
    ax6.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    idx = (p_l_change > .1) & (p_l_change < 1) & (np.abs(p_r_change) <1)
    ax6.plot(np.arange(-30,50),np.nanmean(bigchoicematrix[idx,:],0))
    ax6.set_title('All contrasts - '+wr_name)
    ax6.set_xlabel('Trials relative to block switch')
    ax6.set_ylabel('Average choice')
    ax6.set_ylim(.1,.9)
#%% logistic regression parameters
wr_name = 'FOR01'
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
key = {
       'subject_id':subject_id,
       }   
df_coefficients = pd.DataFrame(behavioranal.SessionFittedChoiceCoefficients() & key)

fig=plt.figure()
ax1=fig.add_axes([0,0,1,.8])
ax2=fig.add_axes([1.2,0,1,.8])
ax3=fig.add_axes([0,-1,1,.8])
ax1.plot(df_coefficients['coefficients_rewards'][df_coefficients['session']>7].mean())
ax2.plot(df_coefficients['coefficients_choices'][df_coefficients['session']>7].mean())
ax3.plot(df_coefficients['score'])
#%%
for rownow in df_coefficients.iterrows():
    ax1.plot(rownow[1]['coefficients_rewards'])
    ax2.plot(rownow[1]['coefficients_choices'])
    

#%% reward ratio over time
wr_name = 'FOR01'
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
key = {
       'subject_id':subject_id,
       }

df_rewardratio = pd.DataFrame(behavioranal.SessionRewardRatio() & key)
rewardratio = list()
sessionrewardratio = list()
sessionstart = list([0])
maxrewardratio = list([0])
for reward_now,maxreward_now in zip(df_rewardratio['session_reward_ratio'],df_rewardratio['session_maximal_reward_ratio'] ):
    rewardratio.extend(reward_now)
    maxrewardratio.extend(maxreward_now)
    sessionrewardratio.append(np.median(reward_now))
    sessionstart.append(len(rewardratio)-1)
rewardratio = np.array(rewardratio)
maxrewardratio = np.array(maxrewardratio)
fig=plt.figure()
ax1=fig.add_axes([0,0,1,.8])
ax1.plot(rewardratio)
ax1.plot(sessionstart,np.ones(len(sessionstart))*.5,'k|',markersize=300)
ax1.plot(maxrewardratio,'--')
ax1.plot(np.ones(len(maxrewardratio))*.35,'r--')
ax2=fig.add_axes([0,-1,1,.8])
ax2.plot(rewardratio/maxrewardratio[:-1])
ax2=fig.add_axes([0,-2,1,.8])
ax2.hist(rewardratio/maxrewardratio[:-1],bins = 20,range = [0, 2])
#%% reward ratio change based on blocks
wr_name = 'FOR04'
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
key = {
       'subject_id':subject_id,
       }
df_blockrewardratio = pd.DataFrame(behavioranal.BlockRewardRatio() & key)
#%
fig=plt.figure()
ax1=fig.add_axes([0,0,1,.8])
ax1.hist(df_blockrewardratio['block_length'],20)
idx = df_blockrewardratio['block_length']>30
ax2=fig.add_axes([0,-1,1,.8])
ax2.hist(np.array(df_blockrewardratio['block_reward_ratio_third_tertile'][idx]-df_blockrewardratio['block_reward_ratio_first_tertile'][idx],dtype=np.float32),40,range = [-1, 1])


#%%
df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key) * (experiment.SessionBlock() & key) * (behavioranal.BlockRewardRatio()&key))
df_behaviortrial['trial_choice_plot'] = np.nan
df_behaviortrial.loc[df_behaviortrial['trial_choice']=='left','trial_choice_plot']=0
df_behaviortrial.loc[df_behaviortrial['trial_choice']=='right','trial_choice_plot']=1

#%
minblocknum = 30
blockchanges=np.where(np.diff(df_behaviortrial['block']))[0]
p_change_L = list()
p_change_R = list()
choices_matrix = list()
for idx in blockchanges:
    prev_blocknum = df_behaviortrial['block_length'][idx]
    next_blocknum = df_behaviortrial['block_length'][idx+1]
    prev_block_p_L = df_behaviortrial['p_reward_left'][idx]
    next_block_p_L = df_behaviortrial['p_reward_left'][idx+1]
    prev_block_p_R = df_behaviortrial['p_reward_right'][idx]
    next_block_p_R = df_behaviortrial['p_reward_right'][idx+1]
    if prev_blocknum > minblocknum and next_blocknum > minblocknum:
        p_change_L.append(float((next_block_p_L-prev_block_p_L)))
        p_change_R.append(float(next_block_p_R-prev_block_p_R))
        choices = np.array(df_behaviortrial['trial_choice_plot'][idx-29:idx+31],dtype=np.float32)
        choices_matrix.append(choices)
choices_matrix = np.asmatrix(choices_matrix)           
change_to_left = np.array(p_change_L) > 0 
change_to_right = np.array(p_change_R) > 0 
choicex_matrix_to_right = choices_matrix[change_to_right,:]
choicex_matrix_to_left = choices_matrix[change_to_left,:]

fig=plt.figure()
ax1=fig.add_axes([0,0,1,.8])
ax1.plot(np.nanmean(choicex_matrix_to_right,0).tolist()[0])
ax1.plot(np.nanmean(choicex_matrix_to_left,0).tolist()[0])

#%% learning curve
wr_name = 'FOR04'
session = 27
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
key = {
       'subject_id':subject_id
       #'session': session
       }
df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key) * (experiment.SessionBlock() & key))
#%
df_behaviortrial['p_reward_sum'] =np.array(df_behaviortrial['p_reward_left']+df_behaviortrial['p_reward_right'],dtype=np.float32)
df_behaviortrial.loc[df_behaviortrial['p_reward_sum']<1,'p_reward_sum'] = .35# df_behaviortrial['p_reward_sum']*1.5
df_behaviortrial.loc[df_behaviortrial['p_reward_sum']>=1,'p_reward_sum'] = df_behaviortrial['p_reward_sum']/2
df_behaviortrial['p_no_reward_sum'] =1-np.array(df_behaviortrial['p_reward_sum'],dtype=np.float32)
df_behaviortrial['reward']=0.
idx = df_behaviortrial['outcome'] == 'hit'
df_behaviortrial.loc[idx , 'reward'] = 1 / df_behaviortrial['p_reward_sum'][idx]
idx = df_behaviortrial['outcome'] == 'miss'
df_behaviortrial.loc[idx , 'reward'] = -1 / (df_behaviortrial['p_no_reward_sum'][idx])
#%
#sessionchange = df_behaviortrial['session'].diff()==1
#%
fig=plt.figure()
ax1=fig.add_axes([0,0,1,.8])
ax1.plot(df_behaviortrial['reward'].cumsum())
#ax1.plot(sessionchange,'|')
#%%
wr_name = 'FOR03'
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
session = 20
block = 3
key = {
       'subject_id':subject_id,
       'session':session
       }

df_behaviorblock = pd.DataFrame(((behavioranal.BlockRewardRatio())))# & key
#%%

fig=plt.figure()
ax1=fig.add_axes([0,0,1,.8])
ax1.hist(df_behaviorblock['block_length'],100)
#%%
windowsize = 40
minperiod = 5
df_behaviortrial['reward']=0
df_behaviortrial['reward'][df_behaviortrial['outcome']=='hit'] = 1
df_behaviortrial['reward'][df_behaviortrial['outcome']=='miss'] = 0
moving_all = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()
df_behaviortrial['reward'][df_behaviortrial['trial_choice']=='left'] = np.nan
moving_right = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()
df_behaviortrial['reward'][df_behaviortrial['outcome']=='hit'] = 1
df_behaviortrial['reward'][df_behaviortrial['outcome']=='miss'] = 0
df_behaviortrial['reward'][df_behaviortrial['trial_choice']=='right'] = np.nan
moving_left = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()


fig=plt.figure()
ax1=fig.add_axes([0,0,1,1])
ax1.plot(moving_all)
ax1.plot(moving_left)
ax1.plot(moving_right)

missnum = sum(df_behaviortrial['outcome']=='miss')
hitnum = sum(df_behaviortrial['outcome']=='hit')
print(hitnum/(hitnum + missnum))

    
#%% download data from datajoint
wr_name = 'FOR02'
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
session = 23
key = {
       'subject_id':subject_id,
       'session':session
       }

df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & key) * (experiment.SessionTrial() & key) * (experiment.SessionBlock()&key) * (behavioranal.TrialReactionTime() & key)))
df_session=pd.DataFrame(experiment.Session() & key)
df_licks=pd.DataFrame((experiment.ActionEvent() & key) * (experiment.BehaviorTrial() & key) * (behavioranal.TrialReactionTime() & key))
df_session
#%% reaction time
df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & key) * (experiment.SessionTrial() & key)  * (behavioranal.TrialReactionTime() & key)))
reaction_times = np.array((df_behaviortrial['reaction_time'][df_behaviortrial['outcome']!='ignore']).values, dtype=np.float32)
fig=plt.figure()
ax1=fig.add_axes([0,0,1,1])
ax1.hist(reaction_times*1000,100,(0,1000))
ax1.set_xlabel('Reaction time (ms)')
ax1.set_ylabel('Trials')
#%% licks on miss trials
misstrials = df_licks['outcome']=='hit'

lick_times_from_first_lick = np.array( df_licks['action_event_time'][misstrials] - df_licks['first_lick_time'][misstrials] , dtype=np.float32)
fig=plt.figure()
ax1=fig.add_axes([0,0,1,1])
ax1.hist(lick_times_from_first_lick*1000,100,(0,1000))
ax1.set_xlabel('Time from first lick(ms)')
ax1.set_ylabel('Lick count')
#%%
df_behaviortrial['trial_choice_plot'] = np.nan
df_behaviortrial['trial_choice_plot'][df_behaviortrial['trial_choice']=='left']=0
df_behaviortrial['trial_choice_plot'][df_behaviortrial['trial_choice']=='right']=1
df_behaviortrial['reward_ratio']=df_behaviortrial['p_reward_right']/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left'])
df_behaviortrial['trial_choice_plot'] = np.nan
df_behaviortrial['trial_choice_plot'][df_behaviortrial['trial_choice']=='left']=0
df_behaviortrial['trial_choice_plot'][df_behaviortrial['trial_choice']=='right']=1
df_behaviortrial['reward_ratio']=df_behaviortrial['p_reward_right']/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left'])
binsize = 5
bias = list()
for idx in range(len(df_behaviortrial)):
    if idx < round(binsize/2) or idx > len(df_behaviortrial)-round(binsize/2):
        bias.append(np.nan)
    else:
        bias_now = np.mean(df_behaviortrial['trial_choice_plot'][idx-round(binsize/2):idx+round(binsize/2)])
        bias.append(bias_now)
        
rewarded = (df_behaviortrial['outcome']=='hit')
unrewarded = (df_behaviortrial['outcome']=='miss')
#%%
fig=plt.figure()
ax1=fig.add_axes([0,0,2,1])
ax1.plot(df_behaviortrial['trial'][rewarded],df_behaviortrial['trial_choice_plot'][rewarded],'k|',color='black',markersize=30,markeredgewidth=2)
ax1.plot(df_behaviortrial['trial'][unrewarded],df_behaviortrial['trial_choice_plot'][unrewarded],'|',color='gray',markersize=15,markeredgewidth=2)
ax1.plot(df_behaviortrial['trial'],bias,'k-')
#ax1.plot(df_behaviortrial['trial'],rewardratio_R / (rewardratio_R+rewardratio_L),'g-')
ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reward_ratio'],'y-')

# =============================================================================
# #%% logistic regression on choices and rewards
# wr_name = 'FOR02'
# subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
# session = 27
# key = {
#        'subject_id':subject_id,
#        'session':session
#        }
# 
# df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & key)))
# #%%
# trials_back = 15
# idx = np.argsort(df_behaviortrial['trial'])
# choices = df_behaviortrial['trial_choice'][idx].values
# choices_digitized = np.zeros(len(choices))
# choices_digitized[choices=='right']=1
# choices_digitized[choices=='left']=-1
# outcomes = df_behaviortrial['outcome'][idx].values
# rewards_digitized = choices_digitized.copy()
# rewards_digitized[outcomes=='miss']=0
# label = list()
# data = list()
# for trial in range(15,len(rewards_digitized)):
#     if choices_digitized[trial] != 0:
#         label.append(choices_digitized[trial])
#         data.append(np.concatenate([rewards_digitized[trial-15:trial],choices_digitized[trial-15:trial]]))
# label = np.array(label)
# data = np.matrix(data)
# #%%
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
# from sklearn.linear_model import LogisticRegression
# logisticRegr = LogisticRegression()
# logisticRegr.fit(x_train, y_train)
# predictions = logisticRegr.predict(x_test)
# score = logisticRegr.score(x_test, y_test)
# print(score)
# 
# fig=plt.figure()
# ax1=fig.add_axes([0,0,2,1])
# coefficients = logisticRegr.coef_
# coefficients = coefficients[0]
# ax1.plot(coefficients)
# coeff_rewards = coefficients[14::-1]
# coeff_choices = coefficients[-1:14:-1]
# =============================================================================

#%%
#%%
df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients() * lab.WaterRestriction())
#%
fig=plt.figure()
ax1=fig.add_axes([0,0,1,1])  
for wr_name in subject_names:
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    idx = df_coeff['subject_id']==subject_id
    if sum(idx) == 1:
        ax1.plot(df_coeff['coefficients_rewards_subject'][idx].values[0])

#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
wr_name = 'FOR03'
subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
session = 27
key = {
       'subject_id':subject_id
       #'session':session
       }

trials_back = 15
first_session = 8
label = list()
data = list()
df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key))
sessions = np.unique(df_behaviortrial_all['session'])
for session in sessions:
    if session >= first_session:
        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
        idx = np.argsort(df_behaviortrial['trial'])
        choices = df_behaviortrial['trial_choice'].values[idx]
        choices_digitized = np.zeros(len(choices))
        choices_digitized[choices=='right']=1
        choices_digitized[choices=='left']=-1
        outcomes = df_behaviortrial['outcome'].values[idx]
        rewards_digitized = choices_digitized.copy()
        rewards_digitized[outcomes=='miss']=0
        for trial in range(trials_back,len(rewards_digitized)):
            if choices_digitized[trial] != 0:
                label.append(choices_digitized[trial])
                data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
label = np.array(label)
data = np.matrix(data)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(x_train, y_train)
#predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)        
coefficients = logisticRegr.coef_
coefficients = coefficients[0]
coeff_rewards = coefficients[trials_back-1::-1]
coeff_choices = coefficients[-1:trials_back-1:-1]
key['coefficients_rewards_subject'] = coeff_rewards
key['coefficients_choices_subject'] = coeff_choices
key['score_subject'] = score