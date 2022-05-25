echo "Generate Phi41 Datasets..."
for N in 1000 10000
do
    for k in 0.0 0.1
    do
        FILE=./data/parabolic_additive_${N}_${k}.npz
        if [ -f "$FILE" ]; then
            echo "$FILE exists."
        else
            echo "Generating $FILE..."
            python parabolic_data.py -N $N -k $k
            echo "$FILE Generated."
        fi
    done
done