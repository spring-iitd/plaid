#!/bin/bash

# python3 injection.py "../data/Car Hacking Dataset/" "benign_data.csv" 0.01

# python train.py "../data/Car Hacking Dataset/" "preprocessed_car_hacking.csv"

# python3 eval.py "../data/Car Hacking Dataset/" 0.01

# python3 train_and_test.py "../data/Car Hacking Dataset/" 0.01

#!/bin/bash

start=0.02
end=0.1
step=0.005

current=$start

while (( $(echo "$current <= $end" | bc -l) )); do
    python3 injection.py "../data/Car Hacking Dataset/" "benign_data.csv" $current
    python3 eval.py "../data/Car Hacking Dataset/" $current
    python3 train_and_test.py "../data/Car Hacking Dataset/" $current
    current=$(echo "$current + $step" | bc -l)
done
