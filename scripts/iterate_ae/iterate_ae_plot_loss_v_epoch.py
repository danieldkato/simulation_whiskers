# -*- coding: utf-8 -*-
"""
Plot various loss terms vs. training epoch for autoencoder/prediction model.

Created on Sun Sep  8 09:15:45 2024

@author: danie
"""

#%% Import statements, setup environment:
import sys
import os
import pathlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import inspect
import socket
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False

hostname = socket.gethostname()

if hostname == 'DESKTOP-PJOJ7HT':
    base = os.path.join('Z:\\', 'users', 'Dan', 'code', 'ws')
elif hostname == 'DESKTOP-1PVCRAF':
    base = os.path.join('E:\\', 'simulation_whiskers')



#%% Define inputs/parameters:

    

# Input parameters:
input_path = os.path.join(base, 'results', 'run734', 'ae_iterate_beta_reconstruction.pickle')

# Define independent variable:
response_var = 'loss_pr' # 'loss_rec_epochs' | 'loss_rec_binned' | 'loss_ce_epochs' | 'loss_sp_epochs' | 'loss_epochs' | 'loss_xor_epochs'

# Define grouping variables; each unique combiation of grouping variable values 
# will correspond to a single curve:
grouping_variables = ['beta_sp', 'beta0', 'beta1', 'beta_xor', 'beta_rec', 'n_hidden']

# Define custom filter if desired:
#flt = lambda x : x.n_hidden==160 and round(x.beta_rec) == round(0)
#flt = lambda x : round(x.beta_sp) == 150 
flt = None

# Plot options:
title_fields = ['n_hidden', 'n_trials_pre', 'beta_rec', 'beta_pr']
infos_per_line = 2

# Output parameters:
save_output = False



#%% Preliminaries:

# Load results:
results = pickle.load(open(input_path, 'rb'))
ae_df = results['ae_df']    

# Apply any filters if requested:
if flt is not None:
    keep = ae_df.apply(flt, axis=1)
    ae_df = ae_df[keep]
    


#%% Generate figure:

if response_var == 'loss_rec_binned':
    loss_terms = [x for x in ae_df.keys() if 'loss_rec' in x]
else:
    loss_terms = [response_var]


# Identify different groups:
groups_df = ae_df[grouping_variables].drop_duplicates()

# Average across runs within each group:
mu_df = ae_df[['epoch'] + grouping_variables + [response_var]]\
    .groupby(['epoch']+grouping_variables).mean().reset_index()

# Iterate over groups:
loss_fig = plt.figure()
for gidx, group in groups_df.iterrows():
    
    # Retrieve results just corresponding to current group:
    curr_grp_results =  pd.merge(ae_df, group.to_frame().T, on=grouping_variables)
    
    # Generate label for current group:
    curr_label = ', '.join(['{}={}'.format(x,group[x]) for x in group.keys()])
    
    # Plot:
    plt.plot(curr_grp_results.epoch, curr_grp_results[response_var], label=curr_label)

# Create title, axis labels, etc:
if response_var == 'loss':
    loss_str = 'Total_loss'
elif response_var == 'loss_rec':
    if 'model_type' in ae_df:
        if len(np.unique(ae_df.model_type)) == 1:
            if ae_df.iloc[0].model_type == 'autoencoder':
                loss_str = 'Reconstruction loss'
            elif ae_df.iloc[0].model_type == 'prediction':
                loss_str = 'Prediction loss'
        else:
            loss_str = 'Reconstruction/prediction loss'
    else:
        loss_str = 'Reconstruction loss'
elif response_var == 'loss_ce':
    loss_str = 'Linear task loss'
elif response_var == 'loss_xor':
    loss_str = 'XOR loss'
elif response_var == 'loss_sp':
    loss_str = 'Sparsity loss'
elif response_var == 'loss_pr':
    loss_str = 'Participation ratio'



plt.xlabel('Training epoch')
plt.ylabel(loss_str)
plt.legend(frameon=False, prop={'size':6})

title_lines = []
title_lines.append('{} vs training epoch'.format(loss_str))

title_clauses = []
for t in title_fields:
    if len(np.unique(ae_df[t])) == 1:
        curr_clause = '{}={}'.format(t, ae_df.iloc[0][t])
        title_clauses.append(curr_clause)
title_linebreaks = np.arange(0, len(title_clauses), infos_per_line)
title_clause_lines = [title_clauses[b:b+3] for b in title_linebreaks]
title_info_lines = [', '.join([y for y in x]) for x in title_clause_lines]

title_lines += title_info_lines
title_str = '\n'.join(title_lines)
plt.title(title_str)



#%% Save output if requested:
    
if save_output:
    
    # Define current output directory:
    input_dir = os.path.split(input_path)[0]
    if 'analysis_metadata' in sys.modules:
        curr_output_dir = increment_dir_name(input_dir, 'loss_plot')
    else: 
        curr_output_dir = os.path.join(input_dir, 'loss_plot')
    
    # Create current output directory if necessary:
    if not os.path.exists(curr_output_dir):
        pathlib.Path(curr_output_dir).mkdir(parents=True,exist_ok=True)
    
    # Save figures:
    fname_loss = response_var.lower().replace(' ', '_')
    plt.figure(loss_fig)
    png_path = os.path.join(curr_output_dir, fname_loss+'.png')
    plt.savefig(png_path)
    
    svg_path = os.path.join(curr_output_dir, fname_loss+'.svg')
    plt.savefig(svg_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(png_path)
        M.add_output(svg_path)
        metadata_path = os.path.join(curr_output_dir, 'plot_{}.json'.format(fname_loss))
        write_metadata(M, metadata_path)