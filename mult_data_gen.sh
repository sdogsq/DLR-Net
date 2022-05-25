echo "Generate MULTIPLICATIVE FORCING Datasets..."
for N in 1000 10000
do
    for k in 0.0 0.1
    do
        FILE=./data/parabolic_multiplicative_${N}_${k}.npz
        if [ -f "$FILE" ]; then
            echo "$FILE exists."
        else
            echo "Generating $FILE..."
            python mult_data.py -N $N -k $k
            echo "$FILE Generated."
        fi
    done
done