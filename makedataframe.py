import pandas as pd

# load left/right thickness, area, volume
lh_th = pd.read_table('lh.aparc.thickness.txt', sep=r'\s+', engine='python')  # first col Subject
rh_th = pd.read_table('rh.aparc.thickness.txt', sep=r'\s+', engine='python')
lh_area = pd.read_table('lh.aparc.area.txt', sep=r'\s+', engine='python')
rh_area = pd.read_table('rh.aparc.area.txt', sep=r'\s+', engine='python')
lh_vol = pd.read_table('lh.aparc.volume.txt', sep=r'\s+', engine='python')
rh_vol = pd.read_table('rh.aparc.volume.txt', sep=r'\s+', engine='python')

aseg = pd.read_table('aseg.volume.txt', sep=r'\s+', engine='python')

# Rename subject column consistently
for df in [lh_th, rh_th, lh_area, rh_area, lh_vol, rh_vol, aseg]:
    df.rename(columns={df.columns[0]:'subject'}, inplace=True)

# Add hemisphere prefix to cortical columns to avoid name collisions
def prefix_cols(df, prefix, skip=['subject']):
    df = df.copy()
    cols = [c for c in df.columns if c not in skip]
    df.rename(columns={c: f'{prefix}_{c}' for c in cols}, inplace=True)
    return df

lh_th = prefix_cols(lh_th, 'lh_th')
rh_th = prefix_cols(rh_th, 'rh_th')
lh_area = prefix_cols(lh_area, 'lh_area')
rh_area = prefix_cols(rh_area, 'rh_area')
lh_vol = prefix_cols(lh_vol, 'lh_vol')
rh_vol = prefix_cols(rh_vol, 'rh_vol')

# Merge all on subject
dfs = [lh_th, rh_th, lh_area, rh_area, lh_vol, rh_vol, aseg]
from functools import reduce
merged = reduce(lambda left,right: pd.merge(left, right, on='subject', how='outer'), dfs)

# Save combined CSV
merged.to_csv('freesurfer_combined_ROIs.csv', index=False)
print('Saved combined to freesurfer_combined_ROIs.csv')


# assume merged includes all timepoints in 'subject' column with names like subjID_timepoint
# split subject into subj and time
merged[['subj','time']] = merged['subject'].str.rsplit('_', n=1, expand=True)

# pivot so each subject has baseline/6m/12m columns for each ROI
rois = [c for c in merged.columns if c not in ['subject','subj','time']]
wide = merged.pivot(index='subj', columns='time', values=rois)

# The pivot creates a MultiIndex for columns: (roi_name, time). Flatten it:
wide.columns = [f'{roi}_{time}' for roi, time in wide.columns]

# example: compute absolute and percent delta for one ROI
roi_example = [c for c in wide.columns if c.startswith('lh_vol_entorhinal')][0]  # pick an example
baseline_col = f'{roi_example}_baseline'  # depends on your time suffix naming
m6_col = f'{roi_example}_6m'
m12_col = f'{roi_example}_12m'

# create delta columns (do for all ROIs programmatically)
for roi in rois:
    b = f'{roi}_baseline'
    m6 = f'{roi}_6m'
    m12 = f'{roi}_12m'
    if b in wide.columns and m6 in wide.columns:
        wide[f'{roi}_delta_6m'] = wide[m6] - wide[b]
        wide[f'{roi}_pctchg_6m']  = 100 * wide[f'{roi}_delta_6m'] / wide[b]
    if b in wide.columns and m12 in wide.columns:
        wide[f'{roi}_delta_12m'] = wide[m12] - wide[b]
        wide[f'{roi}_pctchg_12m']  = 100 * wide[f'{roi}_delta_12m'] / wide[b]

wide.to_csv('freesurfer_wide_with_deltas.csv')
print('Saved wide with deltas to freesurfer_wide_with_deltas.csv')
