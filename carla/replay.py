import time
import can

log_file = "candump-2023-11-16_144841.log"  # Replace with the actual path to your log file

# Function to send CAN messages
def send_can_message(timestamp, vlan_name, data):
    # Create a CAN message
    print(timestamp)
    print(vlan_name)
    print(data)
    message = can.Message(
        arbitration_id=int(vlan_name, 16),
        data=bytearray.fromhex(data)
    )

    # Send the CAN message
    bus.send(message)

# Create a virtual CAN bus
bus = can.interface.Bus(channel='vcan0', bustype='socketcan')

# Read the log file line by line
with open(log_file, 'r') as file:
    # Read the first line to get the initial timestamp
    first_line = file.readline()
    prev_timestamp = float(first_line.split()[0][1:-1])  # Remove parentheses
    file.seek(0)  # Reset file pointer to the beginning

    for line in file:
        # Extract relevant information from the log entry
        timestamp = float(line.split()[0][1:-1])  # Remove parentheses
        vlan_name = line.split()[2]
        data = line.split("[8]")[1].replace(" ", "")

        # Calculate the time difference and sleep
        time_diff = timestamp - prev_timestamp
        time.sleep(time_diff)

        # Call the function to send the CAN message
        send_can_message(timestamp, vlan_name, data)

        # Update the previous timestamp
        prev_timestamp = timestamp

# Close the virtual CAN bus
bus.shutdown()
