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

