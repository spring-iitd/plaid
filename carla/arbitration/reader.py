import can

def send_can_message(complete_data, timestamp, arbitration_id, can_data):
    # print(timestamp, arbitration_id, can_data)
    message = can.Message(
        timestamp=timestamp,
        arbitration_id=int(arbitration_id, 16),
        data=bytearray.fromhex(can_data)
    )
    complete_data.append(message)

def get_data(data):
    complete_data=[]

    for line in data:

        # Extract relevant information from the log entry
        timestamp = float(line.split()[0][1:-1])  
        arbitration_id = line.split()[2]
        can_data = line.split("[8]")[1].replace(" ", "")

        # Call the function to send the CAN message
        send_can_message(complete_data, timestamp, arbitration_id, can_data)
    
    return complete_data
