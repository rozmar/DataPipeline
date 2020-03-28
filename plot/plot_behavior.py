
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import datajoint as dj
from scipy.stats import norm
from pipeline import pipeline_tools, lab, experiment, behavioranal
dj.conn()
import matplotlib.pyplot as plt
import decimal
import warnings
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from matplotlib import cm
#%%
def movingaverage(values, window):
    if window >1:
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'same')
        sma[:round(window/2)] = np.nan
        sma[-round(window/2):] = np.nan
    else:
        sma = values
    return sma

def draw_bs_pairs_linreg(x, y, size=1): 
    """Perform pairs bootstrap for linear regression."""#from serhan aya

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(shape=size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds)) # sampling the indices (1d array requirement)
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps


def merge_dataframes_with_nans(df_1,df_2,basiscol):
    basiscol = 'trial'
    colstoadd = list()
# =============================================================================
#     df_1 = df_behaviortrial
#     df_2 = df_reactiontimes
# =============================================================================
    for colnow in df_2.keys():
        if colnow not in df_1.keys():
            df_1[colnow] = np.nan
            colstoadd.append(colnow)
    for line in df_2.iterrows():
        for colname in colstoadd:
            df_1.loc[df_1[basiscol]==line[1][basiscol],colname]=line[1][colname]
    return df_1
#%%            
def plot_weight_water_early_lick(subjects = None):
    if type(subjects) == str:
        subjects = [subjects]
    df_subject_wr=pd.DataFrame(lab.WaterRestriction() * experiment.Session() * experiment.SessionDetails)
    subject_names_all = df_subject_wr['water_restriction_number'].unique()
    if subjects == None:
        subject_names = subject_names_all
    else: 
        subject_names = list()
        for subject in subjects:
            if subject in subject_names_all:
                subject_names.append(subject)
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
        ax_water.plot(range(1,len(water)+1),water.values)#,linewidth=len(subject_names)+1-i
    ax_water.set_ylabel('Consumed water (ml)')
    ax_water.set_xlabel('Session number')
    #ax_water.set_ylim(0, .6)
    ax_water.legend(subject_names)
    ax_water.set_title('Water consumption over training')

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
        ax_trialnum.plot(range(1,len(trialnums)+1),trialnums)#,linewidth=len(subject_names)+1-i
        ax_earlylick.plot(range(1,len(earlylickrates)+1),earlylickrates)#,linewidth=len(subject_names)+1-i
    ax_trialnum.legend(subject_names)
    ax_trialnum.set_xlabel('Session number')
    ax_trialnum.set_ylabel('Number of trials')
    ax_trialnum.set_title('Number of trials during each session')

    ax_earlylick.legend(subject_names)
    ax_earlylick.set_xlabel('Session number')
    ax_earlylick.set_ylabel('Early lick rate')
    ax_earlylick.set_title('Early lick rate during each session')
   
def plot_block_switches(wr_name = None,sessions = None, probability_change_direction = 1, lickport_num = 2,moving_n = 1,contrast_edges = [0,.1,.2], min_block_length = 30):
    #%%
# =============================================================================
#     wr_name = None
#     
#     sessions = None
#     min_block_length = 6
#     moving_n = 1
#     lickport_num = 2
#     probability_change_direction = 1
#     contrast_edges = [0,.1,.2]
# =============================================================================
    
    
    if probability_change_direction >0:
        multiplier = 1
    else:
        multiplier = -1
    if lickport_num == 2:
        task_protocol = 100
    elif lickport_num == 3:
        task_protocol = 101
    trialnums = np.arange(-30,100)
    
    if wr_name == None:
        key = dict()
        wr_name = 'All subjects'
    else:
        subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
        key ={'subject_id':subject_id}
        
        print(subject_id)
    if sessions == None:
        df_block_starts = pd.DataFrame(behavioranal.SessionBlockSwitchChoices()*behavioranal.SessionTrainingType() & 'session_task_protocol = {}'.format(task_protocol) & 'session > 5' & key)
        session_name = 'all sessions'
    elif len(sessions) == 1:
        df_block_starts = pd.DataFrame(behavioranal.SessionBlockSwitchChoices()*behavioranal.SessionTrainingType() & 'session_task_protocol = {}'.format(task_protocol) & 'session ='+str(sessions[0]) & key)
        session_name = 'session '+str(sessions)
    else:
        df_block_starts = pd.DataFrame(behavioranal.SessionBlockSwitchChoices()*behavioranal.SessionTrainingType() & 'session_task_protocol = {}'.format(task_protocol) & 'session >='+str(np.min(sessions))& 'session <='+str(np.max(sessions)) & key)
        session_name = 'sessions '+str(np.min(sessions))+' to '+str(np.max(sessions))
    
    bigchoicematrix = None
    p_r_change = None
    p_l_change = None
    p_m_change = None
    p_r_next = None
    p_r_prev = None
    p_l_next = None
    p_l_prev = None
    p_m_next = None
    p_m_prev = None
    next_block_length = None
    for idx, line in df_block_starts.iterrows():
        if len(line['p_r_change']) > 0 :
            if bigchoicematrix is None:
                neededblocks = np.asarray(line['block_length_next'])>=min_block_length
                bigchoicematrix = np.asarray(line['choices_matrix'])[neededblocks]  
                p_r_change  = np.asarray(line['p_r_change'])[neededblocks]   
                p_l_change  = np.asarray(line['p_l_change'])[neededblocks]  
                p_m_change  = np.asarray(line['p_m_change'])[neededblocks] 
                next_block_length  = np.asarray(line['block_length_next'])[neededblocks]   
                p_r_next  = np.asarray(line['p_r_next'])[neededblocks]
                p_r_prev  = np.asarray(line['p_r_prev'])[neededblocks]
                p_l_next  = np.asarray(line['p_l_next'])[neededblocks]
                p_l_prev  = np.asarray(line['p_l_prev'])[neededblocks]
                p_m_next  = np.asarray(line['p_m_next'])[neededblocks]
                p_m_prev  = np.asarray(line['p_m_prev'])[neededblocks]
            else:
                neededblocks = np.asarray(line['block_length_next'])>=min_block_length
                bigchoicematrix = np.concatenate((bigchoicematrix,np.asarray(line['choices_matrix'])[neededblocks]))
                p_r_change  =  np.concatenate((p_r_change,np.asarray(line['p_r_change'])[neededblocks]))
                p_l_change  =  np.concatenate((p_l_change,np.asarray(line['p_l_change'])[neededblocks]))
                p_m_change  =  np.concatenate((p_m_change,np.asarray(line['p_m_change'])[neededblocks]))
                next_block_length  =  np.concatenate((next_block_length,np.asarray(line['block_length_next'])[neededblocks]))       
                p_r_next  =  np.concatenate((p_r_next,np.asarray(line['p_r_next'])[neededblocks]))       
                p_r_prev  =  np.concatenate((p_r_prev,np.asarray(line['p_r_prev'])[neededblocks]))   
                p_l_next  =  np.concatenate((p_l_next,np.asarray(line['p_l_next'])[neededblocks]))       
                p_l_prev  =  np.concatenate((p_l_prev,np.asarray(line['p_l_prev'])[neededblocks]))   
                p_m_next  =  np.concatenate((p_m_next,np.asarray(line['p_m_next'])[neededblocks]))       
                p_m_prev  =  np.concatenate((p_m_prev,np.asarray(line['p_m_prev'])[neededblocks]))   
    
    fig=plt.figure()
    ax1=fig.add_axes([0,0,.8,.8])
    ax1.hist([p_r_change,p_l_change,p_m_change],np.arange(0,1,.05),color = ['b','r','g'])
    ylimedges = ax1.get_ylim()
    ax1.set_title('Contrast between blocks - Subject: '+wr_name+' - '+session_name)
    ax1.set_ylabel('Count of block switches')
    ax1.set_xlabel('Change in probability')
    ax1.legend(['right lickport','left lickport','middle lickport'])
    ax1.plot(contrast_edges,np.ones(len(contrast_edges))*np.mean(ylimedges),'k|',markersize = 500)
    ax1.set_xlim(0,1)
    
    ax2=fig.add_axes([1,0,.8,.8])
    ax2.hist(next_block_length,40)
    ax2.set_title('Block length distribution - '+wr_name+' - '+session_name)
    ax2.set_ylabel('# of blocks')
    ax2.set_xlabel('# of trials in each block')
    
    blocknums = list()                   
    ax3=fig.add_axes([0,-1,.8,.8])
    idx = (p_r_change*multiplier <1) & (p_r_change*multiplier >contrast_edges[2]) & (np.abs(p_r_change) <1)
    blocknums.append(sum(idx))
    ax3.plot(trialnums,np.ones(len(trialnums))/lickport_num,'k-')
# =============================================================================
#     ignores = np.isnan(bigchoicematrix[idx,:])
#     ax3.plot(trialnums,movingaverage(np.nanmean(ignores,0),moving_n),'y-')
# =============================================================================
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==1,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax3.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'b-')
    idx = (p_l_change*multiplier >0) & (p_l_change*multiplier >contrast_edges[2]) & (np.abs(p_l_change) <1)
    blocknums.append(sum(idx))
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==0,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax3.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'r-')
    if lickport_num == 3:
        idx = (p_m_change*multiplier >0) & (p_m_change*multiplier >contrast_edges[2]) & (np.abs(p_m_change) <1)
        blocknums.append(sum(idx))
        choicematrix_now = np.asarray(bigchoicematrix[idx,:]==2,float)
        choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
        ax3.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'g-')
    
    ax3.set_title('High contrast - '+wr_name+' - '+session_name +' - {} blocks'.format(blocknums))
    ax3.set_xlabel('Trials relative to block switch')
    ax3.set_ylabel('Average choice')
    if lickport_num == 2:
        ax3.set_ylim(.1,.9)
    elif lickport_num == 3:
        ax3.set_ylim(.1,.5)
    
    blocknums = list()                   
    ax4=fig.add_axes([1,-1,.8,.8])
    idx = (p_r_change*multiplier >contrast_edges[1]) & (p_r_change*multiplier <= contrast_edges[2]) & (np.abs(p_r_change) <1)
    blocknums.append(sum(idx))
    ax4.plot(trialnums,np.ones(len(trialnums))/lickport_num,'k-')
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==1,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax4.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'b-')
    idx = (p_l_change*multiplier >contrast_edges[1]) & (p_l_change*multiplier <= contrast_edges[2]) & (np.abs(p_l_change) <1)
    blocknums.append(sum(idx))
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==0,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax4.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'r-')
    if lickport_num == 3:
        idx = (p_m_change*multiplier >contrast_edges[1]) & (p_m_change*multiplier <= contrast_edges[2]) & (np.abs(p_m_change) <1)
        blocknums.append(sum(idx))
        choicematrix_now = np.asarray(bigchoicematrix[idx,:]==2,float)
        choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
        ax4.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'g-')
    ax4.set_title('Intermediate contrast - '+wr_name+' - '+session_name +' - {} blocks'.format(blocknums))
    ax4.set_xlabel('Trials relative to block switch')
    ax4.set_ylabel('Average choice')
    if lickport_num == 2:
        ax4.set_ylim(.1,.9)
    elif lickport_num == 3:
        ax4.set_ylim(.1,.5)
        
    blocknums = list()    
    ax5=fig.add_axes([0,-2,.8,.8])
    idx = (p_r_change*multiplier >contrast_edges[0]) & (p_r_change*multiplier <= contrast_edges[1]) & (np.abs(p_r_change) <1)
    blocknums.append(sum(idx))
    ax5.plot(trialnums,np.ones(len(trialnums))/lickport_num,'k-')
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==1,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax5.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'b-')
    idx = (p_l_change*multiplier >contrast_edges[0]) & (p_l_change*multiplier <= contrast_edges[1]) & (np.abs(p_l_change) <1)
    blocknums.append(sum(idx))
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==0,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax5.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'r-')
    if lickport_num == 3:
        idx = (p_m_change*multiplier >contrast_edges[0]) & (p_m_change*multiplier <= contrast_edges[1]) & (np.abs(p_m_change) <1)
        blocknums.append(sum(idx))
        choicematrix_now =np.asarray(bigchoicematrix[idx,:]==2,float)
        choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
        ax5.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'g-')
    ax5.set_title('Low contrast - '+wr_name+' - '+session_name +' - {} blocks'.format(blocknums))
    ax5.set_xlabel('Trials relative to block switch')
    ax5.set_ylabel('Average choice')
    if lickport_num == 2:
        ax5.set_ylim(.1,.9)
    elif lickport_num == 3:
        ax5.set_ylim(.1,.5)
    
    blocknums = list()    
    ax6=fig.add_axes([1,-2,.8,.8])
    idx = (p_r_change*multiplier >0) & (p_r_change*multiplier <1) & (np.abs(p_r_change) <1)
    blocknums.append(sum(idx))
    ax6.plot(trialnums,np.ones(len(trialnums))/lickport_num,'k-')
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==1,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax6.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'b-')
    idx = (p_l_change*multiplier > .1) & (p_l_change*multiplier < 1) & (np.abs(p_l_change) <1)
    blocknums.append(sum(idx))
    choicematrix_now = np.asarray(bigchoicematrix[idx,:]==0,float)
    choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
    ax6.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'r-')
    if lickport_num == 3:
        idx = (p_m_change*multiplier > .1) & (p_m_change*multiplier < 1) & (np.abs(p_m_change) <1)
        blocknums.append(sum(idx))
        choicematrix_now = np.asarray(bigchoicematrix[idx,:]==2,float)
        choicematrix_now[np.isnan(bigchoicematrix[idx,:])] =np.nan
        ax6.plot(trialnums,movingaverage(np.nanmean(choicematrix_now,0),moving_n),'g-')
    ax6.set_title('All contrasts - '+wr_name+' - '+session_name +' - {} blocks'.format(blocknums))
    ax6.set_xlabel('Trials relative to block switch')
    ax6.set_ylabel('Average choice')
    if lickport_num == 2:
        ax6.set_ylim(.1,.9)
    elif lickport_num == 3:
        ax6.set_ylim(.1,.5)
    
    #%%
def plotregressionaverage(wr_name = None, sessions = None):
    if wr_name == None:
        key = dict()
        wr_name = 'All subjects'
    else:
        subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
        key ={'subject_id':subject_id}
        
        print(subject_id)
    if sessions == None:
        df_coefficients = pd.DataFrame(behavioranal.SessionFittedChoiceCoefficients() & 'session > 5' & key)
        session_name = 'all sessions'
    elif len(sessions) == 1:
        df_coefficients = pd.DataFrame(behavioranal.SessionFittedChoiceCoefficients() & 'session ='+str(sessions[0]) & key)
        session_name = 'session '+str(sessions)
    else:
        df_coefficients = pd.DataFrame(behavioranal.SessionFittedChoiceCoefficients() & 'session >='+str(np.min(sessions))& 'session <='+str(np.max(sessions)) & key)
        session_name = 'sessions '+str(np.min(sessions))+' to '+str(np.max(sessions))
    
    fig=plt.figure()
    ax1=fig.add_axes([0,0,1,1])  
    ax2=fig.add_axes([0,-1.2,1,1]) 
    for line in df_coefficients.iterrows():
        ax1.plot(line[1]['coefficients_rewards'])
        ax2.plot(line[1]['coefficients_choices'])
    ax1.plot(df_coefficients['coefficients_rewards'].mean(),'k-',linewidth = 4)
    ax2.plot(df_coefficients['coefficients_choices'].mean(),'k-',linewidth = 4)
    ax1.set_xlabel('Trials back')
    ax1.set_ylabel('Reward coeff')
    ax1.set_title('Rewards - subject: ' + wr_name + ' - sessions: ' + session_name)
    ax1.set_ylim([-.5, 1.5])
    ax2.set_xlabel('Trials back')
    ax2.set_ylabel('Choice coeff')
    ax2.set_title('Choices - subject: ' + wr_name + ' - sessions: ' + session_name)
    ax2.set_ylim([-.5, 1.5])
    

def plot_regression_coefficients(plottype = 'NRC',lickportnum = '3lp',subjects = []):
    plt.rcParams.update({'font.size': 15})
    trialstoshow = 15
    #trialstofit = 15
    df_subject_wr=pd.DataFrame(lab.WaterRestriction() * experiment.Session()* experiment.SessionDetails())
    subject_names = df_subject_wr['water_restriction_number'].unique()
    subject_names.sort()
    if len(subjects)>1:
        subjects_real = list()
        for subject_now in subject_names:
            if subject_now in subjects:
                subjects_real.append(subject_now)
        subject_names = subjects_real
    if plottype == 'RNRC':
        if lickportnum == '3lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients3lpRNRC())
        elif lickportnum == '2lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficientsRNRC())
            
    elif plottype == 'RNR':
        if lickportnum == '3lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients3lpRNR())
        elif lickportnum == '2lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficientsRNR())
    elif plottype == 'RC':
        if lickportnum == '3lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients3lpRC())    
        elif lickportnum == '2lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficientsRC())    
        
    elif plottype == 'NRC':
        if lickportnum == '3lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients3lpNRC())    
        elif lickportnum == '2lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficientsNRC())    
        
    elif plottype == 'R':
        if lickportnum == '3lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients3lpR())    
        elif lickportnum == '2lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficientsOnlyRewards())    
        
    elif plottype == 'NR':
        if lickportnum == '3lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients3lpNR())    
        elif lickportnum == '2lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficientsOnlyUnRewardeds())    
        
    elif plottype == 'C':
        if lickportnum == '3lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficients3lpC())    
        elif lickportnum == '2lp':
            df_coeff = pd.DataFrame(behavioranal.SubjectFittedChoiceCoefficientsOnlyChoices())    
    fig=plt.figure()
    axs = list()
    if lickportnum == '3lp':
            sides = ['_right','_left','_middle']
    elif lickportnum == '2lp':
            sides = ['']
    
    for directionidx,direction in enumerate(sides):
        xoffset = 0
        if plottype[0] == 'R':
            ax1=fig.add_axes([0,-directionidx,1,.8])  
            xoffset+=1.2
        if 'NR' in plottype:
            ax2=fig.add_axes([xoffset,-directionidx,1,.8]) 
            xoffset+=1.2
        if 'C' in plottype:
            ax3=fig.add_axes([xoffset,-directionidx,1,.8])
    
        subject_names_legend = list()
        wridxs = list()
        subjectidxes = df_coeff['subject_id']== 'this is not a subject id'
        for wr_name in subject_names:
            subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
            idx = df_coeff['subject_id']==subject_id
            if sum(idx) == 1:
                if plottype[0] == 'R':
                    ax1.plot(range(1,len(df_coeff['coefficients_rewards_subject'+direction].mean())+1),df_coeff['coefficients_rewards_subject'+direction][idx].values[0])
                if 'NR' in plottype:    
                    ax2.plot(range(1,len(df_coeff['coefficients_nonrewards_subject'+direction].mean())+1),df_coeff['coefficients_nonrewards_subject'+direction][idx].values[0])
                if 'C' in plottype:    
                    ax3.plot(range(1,len(df_coeff['coefficients_choices_subject'+direction].mean())+1),df_coeff['coefficients_choices_subject'+direction][idx].values[0])
                subject_names_legend.append(wr_name)
                subjectidxes = subjectidxes | idx
        ax = dict()
        if plottype[0] == 'R':
            ax1.set_xlabel('Choices back')
            ax1.set_ylabel('Coeff')
            ax1.set_title('Rewarded trials'+' - '+direction)
            ax1.legend(subject_names_legend,fontsize='small',loc = 'upper right')
            ax1.plot(range(1,len(df_coeff['coefficients_rewards_subject'+direction].mean())+1),df_coeff['coefficients_rewards_subject'+direction][subjectidxes].mean(),'k-',linewidth = 4)
            ax1.plot([0,len(df_coeff['coefficients_rewards_subject'+direction].mean())],[0,0],'k-')
            ax1.set_xlim([0, trialstoshow])
            ax['ax1'] = ax1
        
        if 'NR' in plottype: 
            ax2.set_xlabel('Choices back')
            ax2.set_ylabel('Coeff')
            ax2.set_title('Unrewarded trials'+' - '+direction)
            ax2.legend(subject_names_legend,fontsize='small',loc = 'upper right')
            ax2.plot(range(1,len(df_coeff['coefficients_nonrewards_subject'+direction].mean())+1),df_coeff['coefficients_nonrewards_subject'+direction][subjectidxes].mean(),'k-',linewidth = 4)
            ax2.plot([0,len(df_coeff['coefficients_nonrewards_subject'+direction].mean())],[0,0],'k-')
            ax2.set_xlim([0, trialstoshow])
            ax['ax2'] = ax2
        if 'C' in plottype:
            ax3.set_xlabel('Choices back')
            ax3.set_ylabel('Coeff')
            ax3.set_title('Choices'+' - '+direction)
            ax3.legend(subject_names_legend,fontsize='small',loc = 'upper right')
            ax3.plot(range(1,len(df_coeff['coefficients_choices_subject'+direction].mean())+1),df_coeff['coefficients_choices_subject'+direction][subjectidxes].mean(),'k-',linewidth = 4)
            ax3.plot([0,len(df_coeff['coefficients_choices_subject'+direction].mean())],[0,0],'k-')
            ax3.set_xlim([0, trialstoshow])
            ax['ax3'] = ax3
    
        axs.append(ax)


    
def plot_reward_rate(wr_name):
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {
           'subject_id':subject_id,
           }

    df_rewardratio = pd.DataFrame(behavioranal.SessionRewardRatio() & key)
    rewardratio = list()
    sessionrewardratio = list()
    sessionstart = list([0])
    maxrewardratio = list([0])
    meanrewardratio = list([0])
    for reward_now,maxreward_now in zip(df_rewardratio['session_reward_ratio'],df_rewardratio['session_maximal_reward_ratio'] ):
        rewardratio.extend(reward_now)
        meanrewardratio.extend(np.ones(len(reward_now))*np.mean(reward_now))
        maxrewardratio.extend(maxreward_now)
        sessionrewardratio.append(np.median(reward_now))
        sessionstart.append(len(rewardratio)-1)
    rewardratio = np.array(rewardratio)
    maxrewardratio = np.array(maxrewardratio)
    fig=plt.figure()
    ax1=fig.add_axes([0,0,2,.8])
    ax1.plot(rewardratio)
    ax1.plot(meanrewardratio,'k--')
    ax1.plot(np.ones(len(meanrewardratio))*.35,'r--')
    ax1.plot(np.ones(len(meanrewardratio))*.41,'r--')
    ax1.plot(sessionstart,np.ones(len(sessionstart))*.5,'k|',markersize=500,markeredgewidth = 3)
    ax1.set_ylim([.1,.6])
    ax1.set_ylabel('actual reward rate')
    ax1.set_xlabel('trial#')
    ax1.set_title(wr_name)
    

def plot_local_psychometric_curve(wr_name = 'FOR08',session = 4, model = 'fitted differential',reward_ratio_binnum = 10):
    
  
# =============================================================================
#     wr_name = 'FOR01'
#     session = 28
#     reward_ratio_binnum = 7
#     model = 'fitted fractional'
# =============================================================================
    
    
    
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {
           'subject_id':subject_id,
           'session': session,
           }
    df_choices = pd.DataFrame((experiment.BehaviorTrial()&key)*(experiment.SessionBlock()&key))
    
    
    if model == 'fitted differential':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataFitted()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_differential_income'][0]
        income_choice = df_local_income['choice_local_differential_income'][0]
        income = df_local_income['local_differential_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveFittedDifferential()& 'subject_id = '+str(subject_id))
    elif model == 'fitted fractional':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataFitted()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_fractional_income'][0]
        income_choice = df_local_income['choice_local_fractional_income'][0]
        income = df_local_income['local_fractional_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveFittedFractional()& 'subject_id = '+str(subject_id))
    elif model == 'boxcar differential':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataBoxCar()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_differential_income'][0]
        income_choice = df_local_income['choice_local_differential_income'][0]
        income = df_local_income['local_differential_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveBoxCarDifferential()& 'subject_id = '+str(subject_id))
    elif model == 'boxcar fractional':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataBoxCar()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_fractional_income'][0]
        income_choice = df_local_income['choice_local_fractional_income'][0]
        income = df_local_income['local_fractional_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveBoxCarFractional()& 'subject_id = '+str(subject_id))
    else:
        print('model not understood')
        
    local_filter  = df_local_income['local_filter'][0]
    reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = behavioranal.bin_psychometric_curve(income,income_choice,reward_ratio_binnum)
    
    
    
    
    

# =============================================================================
#     filter_now = local_filter
#     
#     right_choice = (df_choices['trial_choice'] == 'right').values
#     left_choice = (df_choices['trial_choice'] == 'left').values
#     right_reward = ((df_choices['trial_choice'] == 'right')&(df_choices['outcome'] == 'hit')).values
#     left_reward = ((df_choices['trial_choice'] == 'left')&(df_choices['outcome'] == 'hit')).values
#     
# # =============================================================================
# #     right_choice_conv = np.convolve(right_choice , filter_now,mode = 'valid')
# #     left_choice_conv = np.convolve(left_choice , filter_now,mode = 'valid')
# # =============================================================================
#     right_reward_conv = np.convolve(right_reward , filter_now,mode = 'valid')
#     left_reward_conv = np.convolve(left_reward , filter_now,mode = 'valid')
#     
#     right_choice = right_choice[len(filter_now)-1:]
#     left_choice = left_choice[len(filter_now)-1:]
#     
#     choice_num = np.ones(len(left_choice))
#     choice_num[:]=np.nan
#     choice_num[left_choice] = 0
#     choice_num[right_choice] = 1
#     
#     
# # =============================================================================
# #     right_choice_conv[right_choice_conv==0] = np.nan
# #     left_choice_conv[left_choice_conv==0] = np.nan
# #     reward_ratio_right = right_reward_conv/right_choice_conv
# #     reward_ratio_right[np.isnan(reward_ratio_right)] = 0
# #     reward_ratio_left = left_reward_conv/left_choice_conv
# #     reward_ratio_left[np.isnan(reward_ratio_left)] = 0
# #     reward_ratio_sum = (reward_ratio_right+reward_ratio_left)
# #     reward_ratio_sum[reward_ratio_sum==0]= np.nan
# #     reward_ratio_combined = reward_ratio_right/reward_ratio_sum
# # =============================================================================
#     
#     
#     reward_ratio_sum = (right_reward_conv+left_reward_conv)
#     reward_ratio_sum[reward_ratio_sum==0]= np.nan
#     reward_ratio_combined = right_reward_conv/(reward_ratio_sum)
#     
#     todel = np.isnan(reward_ratio_combined)
#     reward_ratio_combined = reward_ratio_combined[~todel]
#     choice_num = choice_num[~todel]
#     todel = np.isnan(choice_num)
#     reward_ratio_combined = reward_ratio_combined[~todel]
#     choice_num = choice_num[~todel]
#     #%
#     bottoms = np.arange(0,100, 100/reward_ratio_binnum)
#     tops = np.arange(100/reward_ratio_binnum,100.005, 100/reward_ratio_binnum)
#     #%
#     reward_ratio_mean = list()
#     reward_ratio_sd = list()
#     choice_ratio_mean = list()
#     choice_ratio_sd = list()
#     for bottom,top in zip(bottoms,tops):
#         if bottom <0:
#             bottom =0
#         if top > 100:
#             top = 100
#         minval = np.percentile(reward_ratio_combined,bottom)
#         maxval = np.percentile(reward_ratio_combined,top)
#         if minval == maxval:
#             idx = (reward_ratio_combined== minval)
#         else:
#             idx = (reward_ratio_combined>= minval) & (reward_ratio_combined < maxval)
#         reward_ratio_mean.append(np.mean(reward_ratio_combined[idx]))
#         reward_ratio_sd.append(np.std(reward_ratio_combined[idx]))
#         
#         bootstrap = bs.bootstrap(choice_num[idx], stat_func=bs_stats.mean)
#         choice_ratio_mean.append(bootstrap.value)
#         choice_ratio_sd.append(bootstrap.error_width())
# # =============================================================================
# #         
# #         choice_ratio_mean.append(np.mean(choice_num[idx]))
# #         choice_ratio_sd.append(np.std(choice_num[idx]))
# # =============================================================================
# =============================================================================
    fig=plt.figure()
    ax1=fig.add_axes([0,0,.8,.8])
    if 'differential' in model:
        ax1.plot([-.5,.5],[0,1],'k-') 
    else:
        ax1.plot([0,1],[0,1],'k-') 
    ax1.errorbar(reward_ratio_mean,choice_ratio_mean,choice_ratio_sd,reward_ratio_sd,'ko-') 
    if 'differential' in model:
        ax1.set_xlabel('Local differential income')
    else:
        ax1.set_xlabel('Local fractional income')
    
    ax1.set_ylabel('Choice ratio')
    

    right_choice = (df_choices['trial_choice'] == 'right').values
    left_choice = (df_choices['trial_choice'] == 'left').values    
    
    right_p_value = df_choices['p_reward_right'].values
    left_p_value = df_choices['p_reward_left'].values
    sum_p_value = right_p_value + left_p_value
    blockswitch = np.where(np.diff(right_p_value))[0]
    
    ax2=fig.add_axes([1,-1,.8,.8])
    
    ax2.plot(left_choice.cumsum(),right_choice.cumsum(),'k-')
    ax2.plot((left_p_value/sum_p_value).cumsum(),(right_p_value/sum_p_value).cumsum(),'g-')
    ax2.plot(left_choice.cumsum()[blockswitch],right_choice.cumsum()[blockswitch],'ko',markerfacecolor=(0,0, 0, 0),markeredgecolor = (0,0,0,1))
    ax2.plot((left_p_value/sum_p_value).cumsum()[blockswitch],(right_p_value/sum_p_value).cumsum()[blockswitch],'go',markerfacecolor=(0,1, 0, 0),markeredgecolor = (0,1,0,1))

    ax2.set_xlabel('Cumulative left choices')
    ax2.set_ylabel('Cumulative right choices')
    ax2.legend(['Choices','Income ratio'])
    
    ax3=fig.add_axes([0,-1,.8,.8])
    ax3.hist(income)
    if 'differential' in model:
        ax3.set_xlabel('Local differential income')
    else:
        ax3.set_xlabel('Local fractional income')
    ax3.set_ylabel('Number of choices')
    
    ax4= fig.add_axes([1,0,.8,.8])
    ax4.plot([0,1]+list(range(1,len(local_filter)+1))+list(range(len(local_filter),len(local_filter)+2)),np.concatenate([[0,0],local_filter,[0,0]]))
    ax4.set_xlabel('Choices back')
    ax4.set_ylabel('Relative value')
    ax4.set_title('Filter for local reward rate')
 #%%   
def plot_one_session(wr_name = 'FOR02',session = 23, model = 'fitted differential', choice_filter = np.ones(10), local_filter = np.ones(10), RT_filter = np.ones(10), fit = 'not_specified'):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    plt.rcParams.update({'font.size': 18})
#%%

# =============================================================================
#     wr_name = 'FOR09'
#     session = 'last'
#     model = 'fitted fractional'
#     choice_filter = np.ones(20)
#     local_filter = np.ones(10)
#     RT_filter = np.ones(10)
#     fit = 'not_specified'
# =============================================================================

    choice_filter = np.asarray(choice_filter)/sum(choice_filter)
    
    local_filter = np.asarray(local_filter)/sum(local_filter)
    
    RT_filter = np.asarray(RT_filter)/sum(RT_filter)
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    if session == 'last':
        session = np.max((experiment.Session() & 'subject_id = '+str(subject_id)).fetch('session'))
        df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & 'subject_id = '+str(subject_id) & 'session = '+str(session)) * experiment.SessionTrial() * experiment.SessionBlock()* behavioranal.TrialReactionTime).fetch())
        df_reactiontimes = pd.DataFrame((behavioranal.TrialReactionTime() & 'subject_id = '+str(subject_id) & 'session = '+str(session))*behavioranal.TrialLickBoutLenght()*experiment.BehaviorTrial())
        while len(df_behaviortrial)<5:
            session -= 1
            df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & 'subject_id = '+str(subject_id) & 'session = '+str(session)) * experiment.SessionTrial() * experiment.SessionBlock() * behavioranal.TrialReactionTime).fetch())
            df_reactiontimes = pd.DataFrame((behavioranal.TrialReactionTime() & 'subject_id = '+str(subject_id) & 'session = '+str(session))*behavioranal.TrialLickBoutLenght()*experiment.BehaviorTrial())
    else:
        df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & 'subject_id = '+str(subject_id) & 'session = '+str(session)) * experiment.SessionTrial() * experiment.SessionBlock()* behavioranal.TrialReactionTime).fetch())
        df_reactiontimes = pd.DataFrame((behavioranal.TrialReactionTime() & 'subject_id = '+str(subject_id) & 'session = '+str(session))*behavioranal.TrialLickBoutLenght()*experiment.BehaviorTrial())
    
    if (behavioranal.SessionTrainingType() & 'subject_id = ' + str(subject_id) & 'session = ' + str(session)).fetch('session_task_protocol')[0] == 100: #2 lickport
        plottype = '2lickport'
    else:
        plottype = '3lickport'
 
    df_behaviortrial = merge_dataframes_with_nans(df_behaviortrial,df_reactiontimes,'trial')
    if model == 'fitted differential':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataFitted()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_differential_income'][0]
        income_choice = df_local_income['choice_local_differential_income'][0]
        income = df_local_income['local_differential_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveFittedDifferential()& 'subject_id = '+str(subject_id))
    elif model == 'fitted fractional':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataFitted()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_fractional_income'][0]
        income_choice = df_local_income['choice_local_fractional_income'][0]
        income = df_local_income['local_fractional_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveFittedFractional()& 'subject_id = '+str(subject_id))
    elif model == 'boxcar differential':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataBoxCar()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_differential_income'][0]
        income_choice = df_local_income['choice_local_differential_income'][0]
        income = df_local_income['local_differential_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveBoxCarDifferential()& 'subject_id = '+str(subject_id))
    elif model == 'boxcar fractional':
        df_local_income = pd.DataFrame(behavioranal.SessionPsychometricDataBoxCar()& 'subject_id = '+str(subject_id) & 'session = '+str(session))
        income_trialnum = df_local_income['trialnum_local_fractional_income'][0]
        income_choice = df_local_income['choice_local_fractional_income'][0]
        income = df_local_income['local_fractional_income_right'][0]
        df_sigmoid = pd.DataFrame(behavioranal.SubjectPsychometricCurveBoxCarFractional()& 'subject_id = '+str(subject_id))
    else:
        print('model not understood')
    #%
    
    
    sigma = df_sigmoid['sigmoid_fit_sigma'][0]
    mu = df_sigmoid['sigmoid_fit_mu'][0]
    prediction = norm.cdf(income, mu, sigma)
    prediction = np.convolve(prediction,choice_filter,mode = 'same')
    #%
    df_session=pd.DataFrame(experiment.Session() & 'session = '+str(session) & 'subject_id = '+str(subject_id))
    df_behaviortrial['trial_choice_plot'] = np.nan
    df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'left', 'trial_choice_plot'] = 0
    df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'right', 'trial_choice_plot'] = 1
    df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'middle', 'trial_choice_plot'] = .5
    
    
    
    trial_choice_plot_interpolated = df_behaviortrial['trial_choice_plot'].values
    nans, x= np.isnan(trial_choice_plot_interpolated), lambda z: z.nonzero()[0]
    trial_choice_plot_interpolated[nans]= np.interp(x(nans), x(~nans), trial_choice_plot_interpolated[~nans])
    bias = np.convolve(trial_choice_plot_interpolated,choice_filter,mode = 'valid')
    bias = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),bias,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
    if plottype == '2lickport':
        df_behaviortrial['reward_ratio']=df_behaviortrial['p_reward_right']/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left'])
    elif plottype == '3lickport':
        df_behaviortrial['reward_ratio_1']=df_behaviortrial['p_reward_left']/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left']+ df_behaviortrial['p_reward_middle'])
        df_behaviortrial['reward_ratio_2']=(df_behaviortrial['p_reward_left']+df_behaviortrial['p_reward_middle'])/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left']+ df_behaviortrial['p_reward_middle'])
        #%
        leftchoices_filtered = np.convolve(df_behaviortrial['trial_choice'] == 'left',choice_filter,mode = 'valid')
        leftchoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),leftchoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
        rightchoices_filtered = np.convolve(df_behaviortrial['trial_choice'] == 'right',choice_filter,mode = 'valid')
        rightchoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),rightchoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
        middlechoices_filtered = np.convolve(df_behaviortrial['trial_choice'] == 'middle',choice_filter,mode = 'valid')
        middlechoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),middlechoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
        allchoices_filtered = np.convolve(df_behaviortrial['trial_choice'] != 'none',choice_filter,mode = 'valid')
        allchoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),allchoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
        
        #%
        
        
    
    rewarded = (df_behaviortrial['outcome']=='hit')
    unrewarded = (df_behaviortrial['outcome']=='miss')
    
    all_reward = ((df_behaviortrial['outcome'] == 'hit')).values
    all_reward_conv = np.concatenate((np.nan*np.ones(len(local_filter)-1),np.convolve(all_reward , local_filter,mode = 'valid')))
# =============================================================================
#     right_reward = ((df_behaviortrial['trial_choice'] == 'right')&(df_behaviortrial['outcome'] == 'hit')).values
#     left_reward = ((df_behaviortrial['trial_choice'] == 'left')&(df_behaviortrial['outcome'] == 'hit')).values
#     right_reward_conv = np.concatenate((np.nan*np.ones(len(local_filter)-1),np.convolve(right_reward , local_filter,mode = 'valid')))
#     left_reward_conv = np.concatenate((np.nan*np.ones(len(local_filter)-1),np.convolve(left_reward , local_filter,mode = 'valid')))
#     rewardratio_combined = right_reward_conv/(right_reward_conv+left_reward_conv)
# =============================================================================
    
    
    

    fig=plt.figure()
    ax1=fig.add_axes([0,0,2,1])
    
    
# =============================================================================
#     ax1.plot(df_behaviortrial['trial'],rewardratio_combined,'g-',label = 'local reward rate')
# =============================================================================
    
    if plottype == '2lickport':
        ax1.plot(df_behaviortrial['trial'][rewarded],df_behaviortrial['trial_choice_plot'][rewarded],'k|',color='black',markersize=30,markeredgewidth=2)
        ax1.plot(df_behaviortrial['trial'][unrewarded],df_behaviortrial['trial_choice_plot'][unrewarded],'|',color='gray',markersize=15,markeredgewidth=2)
        ax1.plot(df_behaviortrial['trial'],bias,'k-',label = 'choice')
        ax1.plot(income_trialnum,prediction,'g-',label = 'model prediction')
        ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reward_ratio'],'y-')
        ax1.set_yticks((0,1))
        ax1.set_yticklabels(('left','right'))
    elif plottype == '3lickport':
        ax1.stackplot(df_behaviortrial['trial'],  leftchoices_filtered/allchoices_filtered ,  middlechoices_filtered/allchoices_filtered ,  rightchoices_filtered/allchoices_filtered ,colors=['r','g','b'], alpha=0.3 )
        ax1.plot(df_behaviortrial['trial'][rewarded],df_behaviortrial['trial_choice_plot'][rewarded],'k|',color='black',markersize=30,markeredgewidth=2)
        ax1.plot(df_behaviortrial['trial'][unrewarded],df_behaviortrial['trial_choice_plot'][unrewarded],'|',color='gray',markersize=15,markeredgewidth=2)
        ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reward_ratio_1'],'y-')
        ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reward_ratio_2'],'y-')
        ax1.set_yticks((0,.5,1))
        ax1.set_yticklabels(('left','middle','right'))
        ax1.set_ylim([-.1,1.1])
    
    ax1.set_title(wr_name + '   -   session: ' + str(session) + ' - '+str(df_session['session_date'][0]))
    ax1.legend(fontsize='small',loc = 'upper right')
    
    ax2=fig.add_axes([0,-1,2,.8])
    ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_left'],'r-')
    ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_right'],'b-')
    if plottype == '3lickport':
        ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_middle'],'g-')
    ax2.set_ylabel('Reward probability')
    ax2.set_xlabel('Trial #')
    if plottype == '3lickport':
        legenda = ['left','right','middle']
    else:
        legenda = ['left','right']
    ax2.legend(legenda,fontsize='small',loc = 'upper right')
    
    
    
    leftidx_all = (df_behaviortrial['trial_choice'] == 'left')# & (df_behaviortrial['early_lick'] == 'no early')
    rightidx_all = (df_behaviortrial['trial_choice'] == 'right')# & (df_behaviortrial['early_lick'] == 'no early')
    
    leftrts_interpolated = np.ones(len(df_behaviortrial))*np.nan
    leftrts_interpolated[leftidx_all]=df_behaviortrial['reaction_time'][leftidx_all]
    nans, x= np.isnan(leftrts_interpolated), lambda z: z.nonzero()[0]
    leftrts_interpolated[nans]= np.interp(x(nans), x(~nans), leftrts_interpolated[~nans])
    left_RT_conv = np.convolve(leftrts_interpolated,RT_filter,mode = 'valid')
    left_RT_conv = np.concatenate((np.nan*np.ones(int(np.floor((len(RT_filter)-1)/2))),left_RT_conv,np.nan*np.ones(int(np.ceil((len(RT_filter)-1)/2)))))
    
    rightrts_interpolated = np.ones(len(df_behaviortrial))*np.nan
    rightrts_interpolated [rightidx_all]=df_behaviortrial['reaction_time'][rightidx_all]
    nans, x= np.isnan(rightrts_interpolated), lambda z: z.nonzero()[0]
    rightrts_interpolated[nans]= np.interp(x(nans), x(~nans), rightrts_interpolated[~nans])
    right_RT_conv = np.convolve(rightrts_interpolated,RT_filter,mode = 'valid')
    right_RT_conv  = np.concatenate((np.nan*np.ones(int(np.floor((len(RT_filter)-1)/2))),right_RT_conv ,np.nan*np.ones(int(np.ceil((len(RT_filter)-1)/2)))))
    
# =============================================================================
#     leftidx = df_reactiontimes['trial_choice'] == 'left'
#     rightidx = df_reactiontimes['trial_choice'] == 'right'
# =============================================================================
    
    leftboutlengths_interpolated = np.ones(len(df_behaviortrial))*np.nan
    leftboutlengths_interpolated[leftidx_all]=df_behaviortrial['lick_bout_length'][leftidx_all]
    nans, x= np.isnan(leftboutlengths_interpolated ), lambda z: z.nonzero()[0]
    leftboutlengths_interpolated [nans]= np.interp(x(nans), x(~nans), leftboutlengths_interpolated [~nans])
    left_boutlength_conv = np.convolve(leftboutlengths_interpolated ,RT_filter,mode = 'valid')
    left_boutlength_conv = np.concatenate((np.nan*np.ones(int(np.floor((len(RT_filter)-1)/2))),left_boutlength_conv,np.nan*np.ones(int(np.ceil((len(RT_filter)-1)/2)))))
    
    rightboutlengths_interpolated = np.ones(len(df_behaviortrial))*np.nan
    rightboutlengths_interpolated[rightidx_all]=df_behaviortrial['lick_bout_length'][rightidx_all]
    nans, x= np.isnan(rightboutlengths_interpolated  ), lambda z: z.nonzero()[0]
    rightboutlengths_interpolated  [nans]= np.interp(x(nans), x(~nans), rightboutlengths_interpolated  [~nans])
    right_boutlength_conv = np.convolve(rightboutlengths_interpolated  ,RT_filter,mode = 'valid')
    right_boutlength_conv = np.concatenate((np.nan*np.ones(int(np.floor((len(RT_filter)-1)/2))),right_boutlength_conv,np.nan*np.ones(int(np.ceil((len(RT_filter)-1)/2)))))
    
    
    mu = df_sigmoid['sigmoid_fit_mu'][0]
    sigma = df_sigmoid['sigmoid_fit_sigma'][0]
    slope = df_sigmoid['linear_fit_slope'][0]
    c = df_sigmoid['linear_fit_c'][0]
    if (fit == 'not_specified' and 'differential' in model) or 'sigmoid' in fit:
        parameters = {'fit_type':'sigmoid','mu':mu,'sigma':sigma}
    elif (fit == 'not_specified' and 'fractional' in model) or 'linear' in fit:
        parameters = {'fit_type':'linear','slope':slope,'c':c}

    model_performance = behavioranal.calculate_average_likelihood_series(income,income_choice,parameters,local_filter=local_filter)
    
    ax_model=fig.add_axes([0,-2,2,.8])
    ax_model.plot(income_trialnum,model_performance,'k-')
    ax_model.set_xlabel('Trial')
    ax_model.set_ylabel('Average likelihood')
    
    
    
    ax3=fig.add_axes([0,-3,2,.8])
    ax33 = ax3.twinx()
    ax33.plot(df_behaviortrial['trial'],all_reward_conv,'g-',label = 'total reward rate', alpha=0.3)
    
    ax3.plot(df_behaviortrial['trial'][leftidx_all],df_behaviortrial['reaction_time'][leftidx_all],'ro')
    ax3.plot(df_behaviortrial['trial'],left_RT_conv,'r-',linewidth = 3)
    ax3.plot(df_behaviortrial['trial'][leftidx_all],df_behaviortrial['lick_bout_length'][leftidx_all],'rx')
    ax3.plot(df_behaviortrial['trial'],left_boutlength_conv,'r-',linewidth = 1)
    
    ax3.plot(df_behaviortrial['trial'][rightidx_all],df_behaviortrial['reaction_time'][rightidx_all],'bo')
    ax3.plot(df_behaviortrial['trial'],right_RT_conv,'b-',linewidth = 3)
    ax3.plot(df_behaviortrial['trial'][rightidx_all],df_behaviortrial['lick_bout_length'][rightidx_all],'bx')
    ax3.plot(df_behaviortrial['trial'],right_boutlength_conv,'b-',linewidth = 1)
    ax33.set_ylabel('Total reward rate')
    #ax3.set_ylim([0, .5])
    ax3.set_ylabel('Reaction time -o- (s)')
    ax3.set_xlabel('Trial #')
    ax3.set_yscale('log')
    minyval = df_behaviortrial.loc[df_behaviortrial['reaction_time']>=0,'reaction_time'].min()#df_behaviortrial.loc[df_behaviortrial['lick_bout_length']>=0,'lick_bout_length'].min()
    maxyval = np.nanmax([df_behaviortrial.loc[df_behaviortrial['reaction_time']>=0,'reaction_time'].max(),df_behaviortrial.loc[df_behaviortrial['lick_bout_length']>=0,'lick_bout_length'].max()])
    try:
        ax3.set_ylim([float(minyval),float(maxyval)])
    except:
        pass
            
    ax4 = fig.add_axes([0,-4,.8,.8])
    ax4.hist(np.asarray(np.diff(df_behaviortrial['trial_start_time'].values),dtype = 'float'),20)
    ax4.set_xlabel('ITI (s)')
    ax4.set_ylabel('count')
    ax4.set_title('ITI distribution')
    
    
    filterx = np.asarray(range(len(choice_filter)))-np.floor(len(choice_filter)/2)
    filterx = np.concatenate(([filterx[0]-1,filterx[0]],filterx,[filterx[-1],filterx[-1]+1]))
    ax5 = fig.add_axes([1,-4,.8,.8])
    ax5.plot(filterx,np.concatenate(([0,0],choice_filter,[0,0])))
    ax5.set_xlabel('Choices')
    ax5.set_title('Fiter for local choice')
# =============================================================================
#     ax4=fig.add_axes([0,-3,1,.8])                   
#     ax4.plot(rewardratio_R / rewardratio_sum,bias,'ko')        
# =============================================================================
    if plottype == '2lickport':
        plot_local_psychometric_curve(wr_name = wr_name ,session = session, model=model)
        
    
#ax1.set_xlim(00, 600)
#ax2.set_xlim(00, 600)
#%%

def plot_block_based_tuning_curves(wr_name = 'FOR02',minsession = 8,mintrialnum = 20,max_bias = .5,bootstrapnum = 100,only_blocks_above_median = False,only_blocks_above_mean = False,only_blocks_below_mean = False):
    
    #%
    plt.rcParams.update({'font.size': 14})
    
# =============================================================================
#     wr_name = 'FOR10'
#     minsession = 8
#     mintrialnum = 20
#     max_bias = 10
#     bootstrapnum = 50
#     only_blocks_above_median = False
#     only_blocks_above_mean = False,
#     only_blocks_below_mean = False
# =============================================================================
    
    
    max_ignore = 3
    max_p_total = .6

    
    allslopes = list()
    meanslopes = list()
    slopes_ci = list()
    metricnames = ['block_choice_ratio_right']
    metricnames_xaxes = ['block_reward_ratio_right']
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {
           'subject_id':subject_id,
           #'session': session
           }
    df_choice_reward_rate = pd.DataFrame((experiment.SessionBlock()*behavioranal.BlockRewardRatio()*behavioranal.BlockStats()*behavioranal.BlockChoiceRatio()*behavioranal.BlockAutoWaterCount()*behavioranal.SessionBias()*behavioranal.SessionTrainingType()) & key & 'session_task_protocol = 100')
    df_choice_reward_rate = df_choice_reward_rate[(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']) >0]
    #df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_trialnum'] >= mintrialnum]
    #df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_ignores'] <= max_ignore]
    df_choice_reward_rate['biasval'] =df_choice_reward_rate[['session_bias_choice_left','session_bias_choice_right','session_bias_choice_middle']].T.max()# np.abs(df_choice_reward_rate['session_bias_choice']*2 -1)
    
    df_choice_reward_rate['block_relative_value']=df_choice_reward_rate['p_reward_right']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left'])
    df_choice_reward_rate['total_reward_rate']=(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left'])
    needed = (df_choice_reward_rate['total_reward_rate']< 1) & (df_choice_reward_rate['session']>= minsession) & (df_choice_reward_rate['block_choice_ratio_right']>-1) & (df_choice_reward_rate['block_autowater_count']==0) & (df_choice_reward_rate['block_length'] >= mintrialnum) & (df_choice_reward_rate['biasval']<=max_bias) 
    df_choice_reward_rate = df_choice_reward_rate[needed] # unwanted blocks are deleted
    if only_blocks_above_median:
        medianval = df_choice_reward_rate['block_trialnum'].median()
        df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_trialnum'] >= medianval]
    elif only_blocks_above_mean:
        medianval = df_choice_reward_rate['block_trialnum'].mean()
        df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_trialnum'] >= medianval]
    elif only_blocks_below_mean:
        medianval = df_choice_reward_rate['block_trialnum'].mean()
        df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_trialnum'] <= medianval]
    #%
    fig=plt.figure()
    
    ax_blocklenght=fig.add_axes([0,1,.8,.8])
    out = ax_blocklenght.hist(df_choice_reward_rate['block_length'],30)
    ax_blocklenght.set_xlabel('Block length (trials)')
    ax_blocklenght.set_ylabel('Count')
    ax_blocklenght.set_title(wr_name)
    
    
    p_left,p_right,p_middle,block_trialnum,block_reward_rate,block_ignores = np.asarray((experiment.SessionBlock()*behavioranal.BlockStats()& 'subject_id = {}'.format(key['subject_id']) &'session > {}'.format(minsession)).fetch('p_reward_left','p_reward_right','p_reward_middle','block_trialnum','block_reward_rate','block_ignores'),float)
        
    p_left[np.isnan(p_left)]=0
    p_right[np.isnan(p_right)]=0
    p_middle[np.isnan(p_middle)]=0
    p_total = p_left+p_right+p_middle
    needed_blocks = (block_ignores <= max_ignore) & (block_trialnum >= mintrialnum) & (p_total <= max_p_total)
    
    p_total= p_total[needed_blocks]
    block_reward_rate = block_reward_rate[needed_blocks]
    harvesting_efficiency = block_reward_rate/p_total
    ax_efficiency = fig.add_axes([1,1,.8,.8])
    ax_efficiency.hist(harvesting_efficiency)
    ax_efficiency.set_title(wr_name)
    ax_efficiency.set_ylabel('# of blocks')
    ax_efficiency.set_xlabel('Harvesting efficiency')
    
    
    ax_sessionwise_slopes = fig.add_axes([2,1,.8,.8])
    
    
    for idx,(metricname,metricname_x) in enumerate(zip(metricnames,metricnames_xaxes)):#for idx,metricname in enumerate(metricnames):
        relvals = np.sort(df_choice_reward_rate['block_relative_value'].unique())
        choice_ratio_mean = list()
        choice_ratio_sd = list()
        choice_ratio_median = list()
        reward_rate_value = list()
        for relval in relvals:
            choice_rate_vals = df_choice_reward_rate[metricname][df_choice_reward_rate['block_relative_value']==relval]
            choice_ratio_mean.append(choice_rate_vals.mean())
            choice_ratio_median.append(choice_rate_vals.median())
            choice_ratio_sd.append(float(np.std(choice_rate_vals.to_numpy())))
            reward_rate_value.append(float(relval))

        ax_1=fig.add_axes([1,-idx,.8,.8])
        #ax_1.errorbar(reward_rate_value,choice_ratio_mean,choice_ratio_sd,color = 'black',linewidth = 3,marker='o',ms=9)
        ax_1.plot(df_choice_reward_rate[metricname_x],df_choice_reward_rate[metricname],'o',markersize = 3,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
        ax_1.plot([0,1],[0,1],'k-')
        ax_1.set_ylim([0, 1])
        ax_1.set_xlim([0, 1])
        ax_1.set_xlabel('actual relative value (r_R/(r_R+r_L))')
        ax_1.set_ylabel('relative choice (c_R/(c_R+c_L))')
        ax_1.set_title(metricname)
       
        ax_2=fig.add_axes([0,-idx,.8,.8])
        ax_2.errorbar(reward_rate_value,choice_ratio_mean,choice_ratio_sd,color = 'black',linewidth = 3,marker='o',ms=9)
        ax_2.plot(df_choice_reward_rate['block_relative_value'],df_choice_reward_rate[metricname],'o',markersize = 3,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
        ax_2.plot([0,1],[0,1],'k-')
        ax_2.set_ylim([0, 1])
        ax_2.set_xlim([0, 1])
        ax_2.set_xlabel('relative value (p_R/(p_R+p_L))')
        ax_2.set_ylabel('relative choice (c_R/(c_R+c_L))')
        ax_2.set_title(metricname)
        #%
        ax_3=fig.add_axes([2,-idx,.8,.8])
        #%
        xvals = np.asarray(df_choice_reward_rate[metricname_x],dtype = 'float')
        yvals = np.asarray(df_choice_reward_rate[metricname],dtype = 'float')
        todel = (xvals ==1) | (yvals == 1)  | (xvals ==0) | (yvals == 0)
        xvals = xvals[todel==False]
        yvals = yvals[todel==False]
        xvals = xvals/(1-xvals)
        yvals = yvals /(1-yvals)
        #%
        
        xvals = np.log2(xvals)
        yvals = np.log2(yvals)
        todel = (np.isinf(xvals) | np.isinf(yvals)) | (yvals ==0) | (xvals ==0) | (np.isnan(xvals) | np.isnan(yvals))
        xvals = xvals[todel==False]
        yvals = yvals[todel==False]
        try:
            slopes, intercepts = draw_bs_pairs_linreg(xvals, yvals, size=bootstrapnum)
            p = np.polyfit(xvals,yvals,1)
        except:
            slopes = list()
            intercepts = list()
            p = None
        
        #%
        ax_3.plot(xvals,yvals,'o',markersize = 3,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
        ax_3.plot([-3,3],[-3,3],'k-')
        ax_3.plot([-3,3],np.polyval(p,[-3,3]),'r-',linewidth = 3)
        for i in range(bootstrapnum):
            ax_3.plot(np.asarray([-3,3]), slopes[i]*np.asarray([-3,3]) + intercepts[i], linewidth=0.5, alpha=0.2, color='red')
        ax_3.set_xlabel('log reward rate log(r_R/r_L)')
        ax_3.set_ylabel('log choice rate log(c_R/c_L)')
        ax_3.set_title('slope: {:2.2f}, ({:2.2f} - {:2.2f})'.format(np.mean(slopes),np.percentile(slopes, 2.5),np.percentile(slopes, 97.5)))
        allslopes.append(slopes)
        meanslopes.append(np.mean(slopes))
        slopes_ci.append(np.percentile(slopes, [2.5, 97.5]))
        #%
        sessions = df_choice_reward_rate['session'].unique()
        session_slopes = list()
        session_offsets = list()
        for session in sessions:
            sessionidx = df_choice_reward_rate['session']==session
            xvals = np.asarray(df_choice_reward_rate[metricname_x][sessionidx],dtype = 'float')
            yvals = np.asarray(df_choice_reward_rate[metricname][sessionidx],dtype = 'float')
            todel = (xvals ==1) | (yvals == 1)  | (xvals ==0) | (yvals == 0)
            xvals = xvals[todel==False]
            yvals = yvals[todel==False]
            xvals = xvals/(1-xvals)
            yvals = yvals /(1-yvals)
            xvals = np.log2(xvals)
            yvals = np.log2(yvals)
            todel = (np.isinf(xvals) | np.isinf(yvals)) | (yvals ==0) | (xvals ==0) | (np.isnan(xvals) | np.isnan(yvals))
            xvals = xvals[todel==False]
            yvals = yvals[todel==False]
            if len(xvals)>3:
                ax_sessionwise_slopes.plot(xvals,yvals,'o')
                p = np.polyfit(xvals,yvals,1)
                ax_sessionwise_slopes.plot([-3,3],np.polyval(p,[-3,3]),'-',linewidth = 3)
                session_slopes.append(p[0])
                session_offsets.append(p[1])
        ax_sessionwise_slopes.plot([-3,3],np.polyval([np.mean(session_slopes),np.mean(session_offsets)],[-3,3]),'k-',linewidth = 6)
        ax_sessionwise_slopes.set_title('Sessionwise mean slope: {:2.2f} +- {:2.2f}'.format(np.mean(session_slopes),np.std(session_slopes)))
        ax_sessionwise_slopes.set_xlabel('log reward rate log(r_R/r_L)')
        ax_sessionwise_slopes.set_ylabel('log choice rate log(c_R/c_L)')
        #%
    output = {'water_restriction_number':wr_name,
              'metric_names':metricnames,
              'mean_slope':meanslopes,
              'slopes_ci':slopes_ci,
              'bootstrapped_slopes':allslopes,
              'session_wise_slopes':session_slopes,
              'session_wise_offsets':session_offsets,
              'harvesting_efficiency':harvesting_efficiency
              }    
    return output


def plot_block_based_tuning_curves_three_lickports(wr_name = 'FOR09',minsession = 8,mintrialnum = 20,max_bias = .9,bootstrapnum = 100,only_blocks_above_median = False,only_blocks_above_mean = False,only_blocks_below_mean = False,overlay = False):
    #%%
    plt.rcParams.update({'font.size': 14})
    
# =============================================================================
#     wr_name = 'FOR20'
#     minsession = 8
#     mintrialnum = 30
#     max_bias = .9
#     bootstrapnum = 50
#     only_blocks_above_median = False
#     only_blocks_above_mean = False,
#     only_blocks_below_mean = False
#     overlay = True
# =============================================================================
    
    allslopes = list()
    meanslopes = list()
    slopes_ci = list()
    metricnames = ['block_choice_ratio_right','block_choice_ratio_left','block_choice_ratio_middle']
    metricnames_xaxes = ['block_reward_ratio_right','block_reward_ratio_left','block_reward_ratio_middle']
    blockvalues_xaxes = ['block_fractional_value_right','block_fractional_value_left','block_fractional_value_middle']
    plot_colors = ['blue','red','green']
    lickport_names = ['right','left','middle']
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {
           'subject_id':subject_id,
           #'session': session
           }
    df_choice_reward_rate = pd.DataFrame((experiment.SessionBlock()*behavioranal.BlockRewardRatio()*behavioranal.BlockStats()*behavioranal.BlockChoiceRatio()*behavioranal.BlockAutoWaterCount()*behavioranal.SessionBias()*behavioranal.SessionTrainingType()) & key & 'session_task_protocol = 101')
    if len(df_choice_reward_rate)>0:
        df_choice_reward_rate = df_choice_reward_rate[(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']) >0]
        #%
        
        #%
        df_choice_reward_rate['biasval'] =df_choice_reward_rate[['session_bias_choice_left','session_bias_choice_right','session_bias_choice_middle']].T.max()# np.abs(df_choice_reward_rate['session_bias_choice']*2 -1)
        df_choice_reward_rate['block_fractional_value_right']= np.asarray(df_choice_reward_rate['p_reward_right']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        df_choice_reward_rate['block_fractional_value_left']=np.asarray(df_choice_reward_rate['p_reward_left']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        df_choice_reward_rate['block_fractional_value_middle']=np.asarray(df_choice_reward_rate['p_reward_middle']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        df_choice_reward_rate['total_reward_rate']=np.asarray((df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        needed = (df_choice_reward_rate['total_reward_rate']< 1) & (df_choice_reward_rate['session']>= minsession) & (df_choice_reward_rate['block_choice_ratio_right']>-1) & (df_choice_reward_rate['block_autowater_count']==0) & (df_choice_reward_rate['block_length'] >= mintrialnum) & (df_choice_reward_rate['biasval']<=max_bias) 
        df_choice_reward_rate = df_choice_reward_rate[needed] # unwanted blocks are deleted
        if only_blocks_above_median:
            medianval = df_choice_reward_rate['block_trialnum'].median()
            df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_trialnum'] >= medianval]
        elif only_blocks_above_mean:
            medianval = df_choice_reward_rate['block_trialnum'].mean()
            df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_trialnum'] >= medianval]
        elif only_blocks_below_mean:
            medianval = df_choice_reward_rate['block_trialnum'].mean()
            df_choice_reward_rate = df_choice_reward_rate[df_choice_reward_rate['block_trialnum'] <= medianval]
        #%
        fig=plt.figure()
        
        ax_blocklenght=fig.add_axes([0,1,1,.8])
        out = ax_blocklenght.hist(df_choice_reward_rate['block_length'],30)
        ax_blocklenght.set_xlabel('Block length (trials)')
        ax_blocklenght.set_ylabel('Count')
        ax_blocklenght.set_title(wr_name)
        titlesofar=''
        for idx,(metricname,metricname_x,blockvalue,plot_color,lickport_now) in enumerate(zip(metricnames,metricnames_xaxes,blockvalues_xaxes,plot_colors,lickport_names)):#for idx,metricname in enumerate(metricnames):
            if overlay:
                idx = 0
            relvals = np.sort(df_choice_reward_rate[blockvalue].unique())
            choice_ratio_mean = list()
            choice_ratio_sd = list()
            choice_ratio_median = list()
            reward_rate_value = list()
            for relval in relvals:
                choice_rate_vals = df_choice_reward_rate[metricname][df_choice_reward_rate[blockvalue]==relval]
                choice_ratio_mean.append(choice_rate_vals.mean())
                choice_ratio_median.append(choice_rate_vals.median())
                choice_ratio_sd.append(float(np.std(choice_rate_vals.to_numpy())))
                reward_rate_value.append(float(relval))
    
            ax_1=fig.add_axes([1,-idx,.8,.8])
            #ax_1.errorbar(reward_rate_value,choice_ratio_mean,choice_ratio_sd,color = 'black',linewidth = 3,marker='o',ms=9)
            ax_1.plot(df_choice_reward_rate[metricname_x],df_choice_reward_rate[metricname],'o',markersize = 3,alpha = .8, color =plot_color)#,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
            ax_1.plot([0,1],[0,1],'k-')
            ax_1.set_ylim([0, 1])
            ax_1.set_xlim([0, 1])
            ax_1.set_xlabel('actual relative value (r/r all)')
            ax_1.set_ylabel('relative choice (c/c all')
            if overlay:
                ax_1.set_title('Block choice ratio')
            else:
                ax_1.set_title(metricname)
           
            ax_2=fig.add_axes([0,-idx,.8,.8])
            ax_2.errorbar(reward_rate_value,choice_ratio_mean,choice_ratio_sd,color = plot_color,linewidth = 3,marker='o',ms=9)
            ax_2.plot(df_choice_reward_rate[blockvalue],df_choice_reward_rate[metricname],'o',markersize = 3,alpha = .8, color = plot_color)#markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
            ax_2.plot([0,1],[0,1],'k-')
            ax_2.set_ylim([0, 1])
            ax_2.set_xlim([0, 1])
            ax_2.set_xlabel('relative value (p/p all)')
            ax_2.set_ylabel('relative choice (c/c all)')
            if overlay:
                ax_1.set_title('Block choice ratio')
            else:
                ax_1.set_title(metricname)
            #%
            ax_3=fig.add_axes([2,-idx,.8,.8])
            #%
            xvals = np.asarray(df_choice_reward_rate[metricname_x],dtype = 'float')
            yvals = np.asarray(df_choice_reward_rate[metricname],dtype = 'float')
            todel = (xvals ==1) | (yvals == 1)  | (xvals ==0) | (yvals == 0)
            xvals = xvals[todel==False]
            yvals = yvals[todel==False]
            xvals = xvals/(1-xvals)
            yvals = yvals /(1-yvals)
            
            xvals = np.log2(xvals)
            yvals = np.log2(yvals)
            todel = (np.isinf(xvals) | np.isinf(yvals)) | (yvals ==0) | (xvals ==0) | (np.isnan(xvals) | np.isnan(yvals))
            xvals = xvals[todel==False]
            yvals = yvals[todel==False]
            slopes, intercepts = draw_bs_pairs_linreg(xvals, yvals, size=bootstrapnum)
            p = np.polyfit(xvals,yvals,1)
            #%
            ax_3.plot(xvals,yvals,'o',markersize = 3,alpha = .8,color = plot_color)#markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
            ax_3.plot([-4,3],[-4,3],'k-')
            ax_3.plot([-4,3],np.polyval(p,[-4,3]),'-',linewidth = 3,color = plot_color)
            for i in range(bootstrapnum):
                ax_3.plot(np.asarray([-4,3]), slopes[i]*np.asarray([-4,3]) + intercepts[i], linewidth=0.5, alpha=0.2, color=plot_color)
            ax_3.set_xlabel('log reward rate log(r/r rest)')
            ax_3.set_ylabel('log choice rate log(c/c rest)')
            if overlay:
                titlesofar = titlesofar+'\n{} slope: {:2.2f}, ({:2.2f} - {:2.2f})'.format(lickport_now, np.mean(slopes),np.percentile(slopes, 2.5),np.percentile(slopes, 97.5))
                ax_3.set_title(titlesofar)
            else:
                ax_3.set_title('slope: {:2.2f}, ({:2.2f} - {:2.2f})'.format(np.mean(slopes),np.percentile(slopes, 2.5),np.percentile(slopes, 97.5)))
            allslopes.append(slopes)
            meanslopes.append(np.mean(slopes))
            slopes_ci.append(np.percentile(slopes, [2.5, 97.5]))
            #%%
        return metricnames, meanslopes, slopes_ci, allslopes
    else:
        return metricnames, [], [], []
#%%








def plot_block_based_tuning_curves_three_lickports_angular(wr_name = 'FOR09',minsession = 8,mintrialnum = 50,max_bias = 1):
    #%%
    vector_middle = [1,0]
    vector_right = [-1*np.cos(np.pi/3),np.sin(np.pi/3)]
    vector_left = [-1*np.cos(np.pi/3),-1*np.sin(np.pi/3)]
    cmap = cm.get_cmap('jet')
    max_marker_size = 15
    plt.rcParams.update({'font.size': 14})
    
# =============================================================================
#     wr_name = 'HC38'
#     minsession = 8
#     mintrialnum = 50
#     max_bias = 1
# =============================================================================
    
    
    #colors = ['red','black']
    
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {
           'subject_id':subject_id,
           #'session': session
           }
    df_choice_reward_rate = pd.DataFrame((experiment.SessionBlock()*behavioranal.BlockRewardRatio()*behavioranal.BlockStats()*behavioranal.BlockChoiceRatio()*behavioranal.BlockAutoWaterCount()*behavioranal.SessionBias()*behavioranal.SessionTrainingType()) & key & 'session_task_protocol = 101')
    if len(df_choice_reward_rate)>0:
        df_choice_reward_rate = df_choice_reward_rate[(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']) >0]
        #%
        
        #%
        
        df_choice_reward_rate['biasval'] =df_choice_reward_rate[['session_bias_choice_left','session_bias_choice_right','session_bias_choice_middle']].T.max()# np.abs(df_choice_reward_rate['session_bias_choice']*2 -1)
        df_choice_reward_rate['block_fractional_value_right']= np.asarray(df_choice_reward_rate['p_reward_right']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        df_choice_reward_rate['block_fractional_value_left']=np.asarray(df_choice_reward_rate['p_reward_left']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        df_choice_reward_rate['block_fractional_value_middle']=np.asarray(df_choice_reward_rate['p_reward_middle']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        
        #%
        
        df_choice_reward_rate['block_fractional_value_angular_x'] = df_choice_reward_rate['block_fractional_value_right']*vector_right[0] + df_choice_reward_rate['block_fractional_value_middle']*vector_middle[0] +df_choice_reward_rate['block_fractional_value_left']*vector_left[0]
        df_choice_reward_rate['block_fractional_value_angular_y'] = df_choice_reward_rate['block_fractional_value_right']*vector_right[1] + df_choice_reward_rate['block_fractional_value_middle']*vector_middle[1] +df_choice_reward_rate['block_fractional_value_left']*vector_left[1]
        df_choice_reward_rate['block_fractional_value_angular_arc'] = np.arctan2(df_choice_reward_rate['block_fractional_value_angular_y'],df_choice_reward_rate['block_fractional_value_angular_x'])
        df_choice_reward_rate['block_fractional_value_angular_length'] = np.sqrt(df_choice_reward_rate['block_fractional_value_angular_x']**2+df_choice_reward_rate['block_fractional_value_angular_y']**2)
        basevalue = .5/np.cos(np.pi/3-df_choice_reward_rate['block_fractional_value_angular_arc']%(2*np.pi/3))
        df_choice_reward_rate['block_fractional_value_angular_length'] = df_choice_reward_rate['block_fractional_value_angular_length']/basevalue
        
        
        df_choice_reward_rate['block_reward_fraction_angular_x'] = df_choice_reward_rate['block_reward_ratio_right'].astype(float)*vector_right[0] + df_choice_reward_rate['block_reward_ratio_middle'].astype(float)*vector_middle[0] +df_choice_reward_rate['block_reward_ratio_left'].astype(float)*vector_left[0]
        df_choice_reward_rate['block_reward_fraction_angular_y'] = df_choice_reward_rate['block_reward_ratio_right'].astype(float)*vector_right[1] + df_choice_reward_rate['block_reward_ratio_middle'].astype(float)*vector_middle[1] +df_choice_reward_rate['block_reward_ratio_left'].astype(float)*vector_left[1]
        df_choice_reward_rate['block_reward_fraction_angular_arc'] = np.arctan2(df_choice_reward_rate['block_reward_fraction_angular_y'],df_choice_reward_rate['block_reward_fraction_angular_x'])
        df_choice_reward_rate['block_reward_fraction_angular_length'] = np.sqrt(df_choice_reward_rate['block_reward_fraction_angular_x']**2+df_choice_reward_rate['block_reward_fraction_angular_y']**2)
        basevalue = .5/np.cos(np.pi/3-df_choice_reward_rate['block_reward_fraction_angular_arc']%(2*np.pi/3))
        df_choice_reward_rate['block_reward_fraction_angular_length'] = df_choice_reward_rate['block_reward_fraction_angular_length']/basevalue
        
        df_choice_reward_rate['block_choice_fraction_angular_x'] = df_choice_reward_rate['block_choice_ratio_right'].astype(float)*vector_right[0] + df_choice_reward_rate['block_choice_ratio_middle'].astype(float)*vector_middle[0] +df_choice_reward_rate['block_choice_ratio_left'].astype(float)*vector_left[0]
        df_choice_reward_rate['block_choice_fraction_angular_y'] = df_choice_reward_rate['block_choice_ratio_right'].astype(float)*vector_right[1] + df_choice_reward_rate['block_choice_ratio_middle'].astype(float)*vector_middle[1] +df_choice_reward_rate['block_choice_ratio_left'].astype(float)*vector_left[1]
        df_choice_reward_rate['block_choice_fraction_angular_arc'] = np.arctan2(df_choice_reward_rate['block_choice_fraction_angular_y'],df_choice_reward_rate['block_choice_fraction_angular_x'])
        df_choice_reward_rate['block_choice_fraction_angular_length'] = np.sqrt(df_choice_reward_rate['block_choice_fraction_angular_x']**2+df_choice_reward_rate['block_choice_fraction_angular_y']**2)
        basevalue = .5/np.cos(np.pi/3-df_choice_reward_rate['block_choice_fraction_angular_arc']%(2*np.pi/3))
        df_choice_reward_rate['block_choice_fraction_angular_length'] = df_choice_reward_rate['block_choice_fraction_angular_length']/basevalue
        
        df_choice_reward_rate['total_reward_rate']=np.asarray((df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left']+df_choice_reward_rate['p_reward_middle']),float).round(2)
        fig=plt.figure()
        ax_dir=fig.add_axes([1,0,.8,.8])
        ax_magnitude_income=fig.add_axes([1,-.5,.8,.4])
        ax_magnitude_choice=fig.add_axes([.5,-0,.4,.8])
        ax_magnitudes = fig.add_axes([2,0,.8,.8])
        df_choice_reward_rate_original = df_choice_reward_rate
            
            
        needed = (df_choice_reward_rate['total_reward_rate']< 1) & (df_choice_reward_rate['session']>= minsession) & (df_choice_reward_rate['block_choice_ratio_right']>-1) & (df_choice_reward_rate['block_autowater_count']==0) & (df_choice_reward_rate['block_length'] >= mintrialnum) & (df_choice_reward_rate['biasval']<=max_bias)# &(df_choice_reward_rate['block_reward_fraction_angular_length']<.5)
            
    
        df_choice_reward_rate = df_choice_reward_rate_original[needed] # unwanted blocks are deleted
        
        #%
        
        for income_arc,choice_arc,income_ampl,choice_ampl in zip(df_choice_reward_rate['block_reward_fraction_angular_arc'],df_choice_reward_rate['block_choice_fraction_angular_arc'],df_choice_reward_rate['block_reward_fraction_angular_length'],df_choice_reward_rate['block_choice_fraction_angular_length']):
            ax_dir.plot(income_arc,choice_arc,'o',color = cmap(choice_ampl), ms = np.ceil(max_marker_size *income_ampl))
            ax_magnitude_income.plot(income_arc,income_ampl,'o',color = cmap(choice_ampl), ms = np.ceil(max_marker_size *income_ampl))
            ax_magnitude_choice.plot(choice_ampl,choice_arc,'o',color = cmap(choice_ampl), ms = np.ceil(max_marker_size *income_ampl))
            ax_magnitudes.plot(income_ampl,choice_ampl,'o',color = cmap(choice_ampl), ms = np.ceil(max_marker_size *income_ampl))
            
        #ax_dir.plot(df_choice_reward_rate['block_reward_fraction_angular_arc'],df_choice_reward_rate['block_choice_fraction_angular_arc'],'ko',color = color_now)
        ax_dir.set_xlim([-np.pi,np.pi])
        ax_dir.set_ylim([-np.pi,np.pi])
        ax_dir.set_yticks([-np.pi/3*2,0,np.pi/3*2])
        ax_dir.set_xticks([-np.pi/3*2,0,np.pi/3*2])
        ax_dir.set_xticklabels(['L','M','R'])
        ax_dir.set_yticklabels(['L','M','R'])
        ax_dir.set_title(wr_name)
        
        
        #ax_magnitude_income.plot(df_choice_reward_rate['block_reward_fraction_angular_arc'],df_choice_reward_rate['block_reward_fraction_angular_length'],'ko',color = color_now)
        ax_magnitude_income.set_ylim([0,1])
        ax_magnitude_income.set_xticks([-np.pi/3*2,0,np.pi/3*2])
        ax_magnitude_income.set_yticks([0,.5,1])
        ax_magnitude_income.set_xticklabels(['L','M','R'])
        ax_magnitude_income.set_xlim([-np.pi,np.pi])
        ax_magnitude_income.set_xlabel('Income direction')
        ax_magnitude_income.set_ylabel('Income amplitude')
        
        
        #ax_magnitude_choice.plot(df_choice_reward_rate['block_choice_fraction_angular_length'],df_choice_reward_rate['block_choice_fraction_angular_arc'],'ko',color = color_now)
        ax_magnitude_choice.set_xlim([0,1])
        ax_magnitude_choice.set_yticks([-np.pi/3*2,0,np.pi/3*2])
        ax_magnitude_choice.set_yticklabels(['L','M','R'])
        ax_magnitude_choice.set_ylim([-np.pi,np.pi])
        ax_magnitude_choice.set_xticks([0,.5, 1])
        ax_magnitude_choice.set_xlabel('Choice amplitude')
        ax_magnitude_choice.set_ylabel('Choice direction')
        
        
        #ax_magnitudes.plot(df_choice_reward_rate['block_reward_fraction_angular_length'],df_choice_reward_rate['block_choice_fraction_angular_length'],'ko',color = color_now)
        ax_magnitudes.set_xlim([0,1])
        ax_magnitudes.set_ylim([0,1])
        ax_magnitudes.set_xlabel('Income amplitude')
        ax_magnitudes.set_ylabel('Choice amplitude')



#%%

def plot_tuning_curve_change_during_block(wr_name = 'FOR02',minsession = 8,mintrialnum = 20,max_bias = .5,bootstrapnum = 100):# TODO: FINISH this one
    #%%
# =============================================================================
#     wr_name = 'FOR01'
#     minsession = 8
#     mintrialnum = 30
#     max_bias = .5
#     bootstrapnum = 100
# =============================================================================
    allslopes = list()
    meanslopes = list()
    slopes_ci = list()
    
    
    subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
    key = {
           'subject_id':subject_id,
           #'session': session
           }
    df_choice_reward_rate = pd.DataFrame((experiment.SessionBlock()*behavioranal.BlockRewardRatio()*behavioranal.BlockStats()*behavioranal.BlockChoiceRatio()*behavioranal.BlockAutoWaterCount()*behavioranal.SessionBias()) & key )
    #%
    
    #%
    df_choice_reward_rate['biasval'] = np.abs(df_choice_reward_rate['session_bias_choice']*2 -1)
    
    df_choice_reward_rate['block_relative_value']=df_choice_reward_rate['p_reward_right']/(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left'])
    df_choice_reward_rate['total_reward_rate']=(df_choice_reward_rate['p_reward_right']+df_choice_reward_rate['p_reward_left'])
    needed = (df_choice_reward_rate['total_reward_rate']<= 1) & (df_choice_reward_rate['session']>= minsession) & (df_choice_reward_rate['block_choice_ratio']>-1) & (df_choice_reward_rate['block_autowater_count']==0) & (df_choice_reward_rate['block_length'] >= mintrialnum) & (df_choice_reward_rate['biasval']<=max_bias)
    df_choice_reward_rate = df_choice_reward_rate[needed] # unwanted blocks are deleted
    
    
    rewardratios = np.stack(df_choice_reward_rate.block_reward_ratios_incremental.values)
    choiceratios = np.stack(df_choice_reward_rate.block_choice_ratios_incremental.values)
    
    #%%
    
    fig=plt.figure()
    
    ax_blocklenght=fig.add_axes([0,1,1,.8])
    out = ax_blocklenght.hist(df_choice_reward_rate['block_length'],30)
    ax_blocklenght.set_xlabel('Block length (trials)')
    ax_blocklenght.set_ylabel('Count')
    ax_blocklenght.set_title(wr_name)
    for idx,(metricname,metricname_x) in enumerate(zip(metricnames,metricnames_xaxes)):#for idx,metricname in enumerate(metricnames):
        relvals = np.sort(df_choice_reward_rate['block_relative_value'].unique())
        choice_ratio_mean = list()
        choice_ratio_sd = list()
        choice_ratio_median = list()
        reward_rate_value = list()
        for relval in relvals:
            choice_rate_vals = df_choice_reward_rate[metricname][df_choice_reward_rate['block_relative_value']==relval]
            choice_ratio_mean.append(choice_rate_vals.mean())
            choice_ratio_median.append(choice_rate_vals.median())
            choice_ratio_sd.append(float(np.std(choice_rate_vals.to_numpy())))
            reward_rate_value.append(float(relval))

        ax_1=fig.add_axes([1,-idx,.8,.8])
        #ax_1.errorbar(reward_rate_value,choice_ratio_mean,choice_ratio_sd,color = 'black',linewidth = 3,marker='o',ms=9)
        ax_1.plot(df_choice_reward_rate[metricname_x],df_choice_reward_rate[metricname],'o',markersize = 3,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
        ax_1.plot([0,1],[0,1],'k-')
        ax_1.set_ylim([0, 1])
        ax_1.set_xlim([0, 1])
        ax_1.set_xlabel('actual relative value (r_R/(r_R+r_L))')
        ax_1.set_ylabel('relative choice (c_R/(c_R+c_L))')
        ax_1.set_title(metricname)
       
        ax_2=fig.add_axes([0,-idx,.8,.8])
        ax_2.errorbar(reward_rate_value,choice_ratio_mean,choice_ratio_sd,color = 'black',linewidth = 3,marker='o',ms=9)
        ax_2.plot(df_choice_reward_rate['block_relative_value'],df_choice_reward_rate[metricname],'o',markersize = 3,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
        ax_2.plot([0,1],[0,1],'k-')
        ax_2.set_ylim([0, 1])
        ax_2.set_xlim([0, 1])
        ax_2.set_xlabel('relative value (p_R/(p_R+p_L))')
        ax_2.set_ylabel('relative choice (c_R/(c_R+c_L))')
        ax_2.set_title(metricname)
        #%
        ax_3=fig.add_axes([2,-idx,.8,.8])
        #%
        xvals = np.asarray(df_choice_reward_rate[metricname_x],dtype = 'float')
        yvals = np.asarray(df_choice_reward_rate[metricname],dtype = 'float')
        todel = (xvals ==1) | (yvals == 1)  | (xvals ==0) | (yvals == 0)
        xvals = xvals[todel==False]
        yvals = yvals[todel==False]
        xvals = xvals/(1-xvals)
        yvals = yvals /(1-yvals)
        #%
        
        xvals = np.log2(xvals)
        yvals = np.log2(yvals)
        todel = (np.isinf(xvals) | np.isinf(yvals)) | (yvals ==0) | (xvals ==0) | (np.isnan(xvals) | np.isnan(yvals))
        xvals = xvals[todel==False]
        yvals = yvals[todel==False]
        slopes, intercepts = draw_bs_pairs_linreg(xvals, yvals, size=bootstrapnum)
        p = np.polyfit(xvals,yvals,1)
        #%
        ax_3.plot(xvals,yvals,'o',markersize = 3,markerfacecolor = (.5,.5,.5,1),markeredgecolor = (.5,.5,.5,1))
        ax_3.plot([-3,3],[-3,3],'k-')
        ax_3.plot([-3,3],np.polyval(p,[-3,3]),'r-',linewidth = 3)
        for i in range(bootstrapnum):
            ax_3.plot(np.asarray([-3,3]), slopes[i]*np.asarray([-3,3]) + intercepts[i], linewidth=0.5, alpha=0.2, color='red')
        ax_3.set_xlabel('log reward rate log(r_R/r_L)')
        ax_3.set_ylabel('log choice rate log(c_R/c_L)')
        ax_3.set_title('slope: {:2.2f}, ({:2.2f} - {:2.2f})'.format(np.mean(slopes),np.percentile(slopes, 2.5),np.percentile(slopes, 97.5)))
        allslopes.append(slopes)
        meanslopes.append(np.mean(slopes))
        slopes_ci.append(np.percentile(slopes, [2.5, 97.5]))
    #return metricnames, meanslopes, slopes_ci, allslopes