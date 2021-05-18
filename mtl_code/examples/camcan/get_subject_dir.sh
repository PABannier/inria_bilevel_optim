for patient_dir in $(ls stcs)
do 
    patient_id="${patient_dir:4}"
    
    echo "================================="
    echo "GETTING SUBJECT_DIR FOR $patient_id"
    echo "================================="

    scp -r drago3:/storage/store/data/camcan-mne/freesurfer/$patient_id subjects_dir
done
