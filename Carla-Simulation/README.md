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

## Step 3: Enable Virtual CAN Bus (vcan0)
```bash
# Load the vcan module and set up vcan0
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

## Step 4: Move Required Files

Before running the simulation, move the `bighonda.dbc`,`log_gen.py`, `plot_graphs.py` and `run_simulation.sh` files from their current directory to the `~/carla_0.9.15/PythonAPI/examples/` directory.

```bash
# Move files from the current directory to the appropriate directory
mv bighonda.dbc log_gen.py plot_graphs.py run_simulation.sh ~/carla_0.9.15/PythonAPI/examples/
```

Move `custom_agent.py` to the `~/carla_0.9.15/PythonAPI/carla/agents/navigation/` directory.

```bash
# Move custom_agent.py to the navigation directory
mv custom_agent.py ~/carla_0.9.15/PythonAPI/carla/agents/navigation/
```

## Step 5: Run the Simulation

To execute the simulation, run the `run_simulation.sh` script.

1. **Create Required Directories**
Before running the simulation, create the `Logs` and `Graphs` folders to store the generated logs and graphs.

```bash
# Create Logs and Graphs directories
mkdir -p Logs Graphs
```

2. **Run the Simulation**
```bash
# Grant execute permission to the script
chmod +x run_simulation.sh

# Execute the script
./run_simulation.sh
```

## Step 6: Run the Attack Simulation

To run the attack simulation, modify `log_gen.py` by uncommenting line **839** and indenting the next two lines properly.

1. **Modify `log_gen.py` for Attack Simulation**

Uncomment line 839 and indent the next two lines:
```bash
# Original Code (Commented)
# if current_timestamp < 22.6015625 or current_timestamp > 22.7265625:
world_1.player.apply_control(control_1)
gen_control_obj_1.append(control_1)
world_2.player.apply_control(control_2)

# Modified Code (Uncomment and indent correctly)
if current_timestamp < 22.6015625 or current_timestamp > 22.7265625:
    world_1.player.apply_control(control_1)
    gen_control_obj_1.append(control_1)
world_2.player.apply_control(control_2)
```

2. **Modify Time Interval for Different Attack Scenarios**

If you want to perform the attack at a different time interval, modify the range in the following line:
```bash
if current_timestamp < 22.6015625 or current_timestamp > 22.7265625:
```

3. **Save and Exit**

4. **Run the simulation**
```bash
./run_simulation.sh
```
