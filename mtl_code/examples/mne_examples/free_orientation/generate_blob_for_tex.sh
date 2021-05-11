# SAMPLE DATASET
conditions=("Left Auditory" "Right Auditory" "Left visual" "Right visual")

for i in "${conditions[@]}"
do
    echo "================================="
    echo "GENERATING DATA FOR $i"
    echo "================================="

    python plot_for_tex.py --condition $i --dataset sample
done


# SOMATO DATASET

echo "=========================="
echo "GENERATING DATA FOR SOMATO"
echo "==========================" 

python plot_for_tex.py --dataset somato

