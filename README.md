# supra

### Setup: 
FreeSurfer is installed and on your PATH.

You have run recon-all successfully for every subject and each timepoint you want to analyze. (Check SUBJECTS_DIR/<subj>/scripts/recon-all.log for failures.)

SUBJECTS_DIR environment variable is set to the directory containing all subject folders.

Create a plain text file subj_list.txt with one FreeSurfer subject ID per line (for the set of images you want to aggregate). Example:

subj001_baseline
subj002_baseline
subj001_6m
subj002_6m
...


Include timepoint in the subject ID if you ran recon-all separately per timepoint (recommended).

2) What each command does (brief)

aparcstats2table --hemi lh --meas thickness --parc aparc ...
→ Produces a table where each row is a subject and each column is the mean cortical thickness (in millimeters) for a Desikan region in the left hemisphere.

aparcstats2table --hemi rh --meas thickness --parc aparc ...
→ Same but for the right hemisphere.

aparcstats2table --hemi lh --meas area --parc aparc ...
→ Table with surface area (units: mm²) per left-hemisphere Desikan region.

aparcstats2table --hemi rh --meas area --parc aparc ...
→ Surface area for right hemisphere.

aparcstats2table --hemi lh --meas volume --parc aparc ...
→ Cortical volume (mm³) per left-hemisphere Desikan region.

aparcstats2table --hemi rh --meas volume --parc aparc ...
→ Cortical volume for right hemisphere.

asegstats2table --subjectsfile subj_list.txt --meas volume --tablefile aseg.volume.txt
→ Table of subcortical volumes (mm³) for structures in the FreeSurfer aseg (thalamus, caudate, putamen, hippocampus, brainstem, etc.). This file often includes a column for estimated total intracranial volume (ICV/eTIV) as well (check header).

All tables are simple delimited text files with subject IDs in the first column and region names in the header row.
## 1 
Run the commandline.txt file in the command line while in the FreeSurfer $SUBJECTS_DIR (one-liners per measure)
