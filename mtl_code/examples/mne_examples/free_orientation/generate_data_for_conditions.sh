estimators=(lasso-cv lasso-sure adaptive-cv adaptive-sure)
conditions=("Left Auditory" "Right Auditory" "Left visual" "Right visual")

for i in "${conditions[@]}"
do
    echo "================================="
    echo "GENERATING DATA FOR $i"
    echo "================================="

    for j in "${estimators[@]}"
    do
        python generate_data.py --estimator $j --condition $i --dataset sample
    done
done


# SOMATO DATASET

echo "=========================="
echo "GENERATING DATA FOR SOMATO"
echo "==========================" 

for i in "${estimators[@]}"
do 
    python generate_data.py --estimator $i --dataset somato
done
