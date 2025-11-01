fmriprep-docker /ssd/gaojianxiong/CineBrain/dcm2bids/BIDS /ssd/gaojianxiong/fmriprep/BIDS2 participant --skip_bids_validation \
 --participant-label 0001 -w /ssd/gaojianxiong/fmriprep/fmriprep_BIDS2/tmp --nthreads 64 --omp-nthreads 64 \
 --output-spaces MNI152NLin6Asym:res-2 fsaverage5 --cifti-output --use-aroma --ignore slicetiming sbref t2w \
 --fs-subjects-dir /ssd/gaojianxiong/fmriprep/Test/freesurfer --fs-license-file /ssd/gaojianxiong/license.txt