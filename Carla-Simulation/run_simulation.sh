#!/bin/bash

log_file="./Logs/can_data_logs.log"

commands=()
timestamps=()

clearTmpFiles() {
    rm -rf ./Logs/*.log
    rm -rf ./Graphs/*.png
}

startCanDump() {
    candump -tz vcan0 > ./Logs/can_data_logs.log &
    sleep 1s
}

runGenLogScript() {
    python3 ./log_gen.py
}

killCanDump() {
    pkill -9 candump
}

postProcessLogs() {
    sed -i 's/^ //' "$log_file"
}

main() {
    clearTmpFiles
    startCanDump
    runGenLogScript
    killCanDump
    postProcessLogs
}

main
