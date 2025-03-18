## CARLA Simulation Setup for Automotive Security Platform

This guide will help you set up CARLA and run the custom simulation designed for security research and intrusion detection.

## Step 1: Install CARLA

Follow the official CARLA Quickstart Guide to install CARLA as a package:

🔗 [CARLA Quickstart Guide](https://carla.readthedocs.io/en/latest/start_quickstart/)

## Step 2: Create and Configure Python Virtual Environment

1. **Set up virtual environment:**
```bash
# Create a virtual environment
cd ~/carla_0.9.15
python3 -m venv carla-venv

# Activate virtual environment
source carla-venv/bin/activate
```

2. **Install required Python packages:**
```bash
# Upgrade pip3 to the latest version
pip3 install --upgrade pip

# Install all packages from requirements.txt
pip3 install -r requirements.txt
```

3. **Enable Virtual CAN Bus (vcan0)**
```bash
# Load the vcan module and set up vcan0
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

4. **Move Required Files**

Before running the simulation, move the `log_gen.py` and `run_simulation.sh` files to the `~/carla_0.9.15/PythonAPI/examples/` directory.

```bash
# Move files to the appropriate directory
mv log_gen.py run_simulation.sh ~/carla_0.9.15/PythonAPI/examples/
```

5. **Run the Simulation**

To execute the simulation, run the `run_simulation.sh` script.

5.1. **Create Required Directories**
Before running the simulation, create the `Logs` and `Graphs` folders to store the generated logs and graphs.

```bash
# Create Logs and Graphs directories
mkdir -p Logs Graphs
```

5.2. **Run the Simulation**
```bash
# Grant execute permission to the script
chmod +x run_simulation.sh

# Execute the script
./run_simulation.sh
```
