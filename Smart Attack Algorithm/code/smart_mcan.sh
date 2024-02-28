#!/bin/bash

# python3 injection.py "../data/M-CAN Intrusion Dataset/" "g80_mcan_normal_data.csv"

# python train.py "../data/M-CAN Intrusion Dataset/" "preprocessed_mcanss.csv"

# python3 eval.py "../data/M-CAN Intrusion Dataset/"

# python3 train_and_test.py "../data/M-CAN Intrusion Dataset/"


start=0.022
end=0.04
step=0.002

current=$start

while (( $(echo "$current <= $end" | bc -l) )); do
    python3 injection.py "../data/M-CAN Intrusion Dataset/" "g80_mcan_normal_data.csv" $current
    python3 eval.py "../data/M-CAN Intrusion Dataset/" $current
    python3 train_and_test.py "../data/M-CAN Intrusion Dataset/" $current
    current=$(echo "$current + $step" | bc -l)
done
