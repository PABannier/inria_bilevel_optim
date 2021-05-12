conditions=("Left Auditory" "Right Auditory" "Left visual" "Right visual")

for i in "${conditions[@]}"
do
    echo "================================="
    echo "GENERATING DATA FOR $i"
    echo "================================="

    python generate_data.py --condition $i --dataset sample
done


# SOMATO DATASET

echo "=========================="
echo "GENERATING DATA FOR SOMATO"
echo "==========================" 

python generate_data.py --dataset somato

