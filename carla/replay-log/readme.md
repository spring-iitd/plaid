# CARLA Log Replay

This project provides tools to replay logs on a CARLA simulator server. It allows users to visualize and analyze vehicle behavior by replaying previously recorded logs.

## File Descriptions

- `biglog.log`: A primary log file with large precision that contains recorded data from a CARLA simulation.
- `fixeddelta.log`: A log file with recorded data at fixed delta intervals.
- `log_play.py`: A script to initiate the log replay on the CARLA server.
- `max_honda.dbc`: A modified DBC file with large precision containing the CAN (Controller Area Network) message descriptions specific to the Max Honda vehicle model.
- `reader.py`: (helper file) A script responsible for reading and processing the log files.
- `replay_log.py`: A script to handle the replaying of log data, converting it into a format that CARLA can understand and replay.
- `readme.md`: (This file) Documentation for understanding and using the project components.

## Prerequisites

- [CARLA Simulator](https://carla.org/)
- Python 3.6+
- Required Python packages (install using `pip install -r requirements.txt`)

## Setup

1. Clone this repository to your local machine:

   ```bash

   git clone <repository-url>

   cd <repository-directory>

   ```
2. Install the required Python packages:

   ```bash

   pip install -r requirements.txt

   ```
3. Ensure that your CARLA simulator server is up and running.

## Usage

1. **Replaying a Log**

   To replay a specific log, use the `log_play.py` script. Not useful as of now. For example, to replay `biglog.log`, run:

   ```bash

   python3 log_play.py

   ```
2. **Using the `replay_log.py` Script**

   Alternatively, you can use `replay_log.py`. This script can be used play simulation which internally uses reader.py to read log

   ```bash

   python replay_log.py

   ```
3. **Reading a Log File**

   To read the contents of a log file using `reader.py`, execute:

   ```bash

   python reader.py --log_file biglog.log

   ```

## Additional Information

- The `max_honda.dbc` file contains the necessary CAN message descriptions. If you are working with different vehicle models, make sure to use the correct DBC file.
- Make sure to configure the log file paths accurately when running the scripts.

## Contributing

We welcome contributions! Please fork the repository and create a pull request for any bug fixes, feature enhancements, or new functionalities.
