def calculate_crc(data):
    """
    Calculate CRC-15 checksum for the given data.

    Args:
        data (str): Binary data string.

    Returns:
       CRC-15 checksum.

    """
    crc = 0x0000

    # CRC-15 polynomial
    poly = 0x4599

    for bit in data:
        # XOR with the current bit shifted left by 14 bits
        crc ^= (int(bit) & 0x01) << 14

        for _ in range(15):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1

        # Ensuring 15 bits
        crc &= 0x7FFF

    return crc

def stuff_bits(binary_string):
    """
    Inserting '1' after every 5 consecutive '0's in the binary string.

    Args:
        binary_string (str): Binary string to be stuffed.

    Returns:
        str: Binary string after stuffing.

    """
    result = ''

    # Initialize a count for consecutive 0's
    count = 0

    for bit in binary_string:

        # Appending the current bit to the result string
        result += bit
        
        # Incrementing the count if the current bit is 0
        if bit == '0':
            count += 1
            
            # Inserting a 1 after 5 consecutive 0's
            if count == 5:
                result += '1'
                # Reseting the count after inserting the 1
                count = 0
        else:
            # Reseting the count if the current bit is not 0
            count = 0

    return result


def hex_to_bits(hex_value, num_bits):
    """
    Convert hexadecimal value to binary string with specified number of bits.
    
    Args:
        hex_value (str): Hexadecimal value to be converted.
        num_bits (int): Number of bits for the resulting binary string.

    Returns:
        str: Binary string representation of the hexadecimal value.
    """
    return bin(int(hex_value, 16))[2:].zfill(num_bits)


def frame_len(can_id, dlc, data):
    """ 
    Converting CAN frame components to a binary string according to the CAN protocol.

    Args:
        can_id (str): CAN identifier in hexadecimal format.
        dlc (int): Data Length Code indicating the number of bytes of data.
        data (list): List of hexadecimal bytes representing data.

    Returns:
        int: Length of binary string representing the formatted CAN frame.

    """

    # Start of Frame (SOF) bit
    start_of_frame = '0'
 
    # Converting CAN identifier to 11-bit binary representation
    can_id_bits = hex_to_bits(can_id, 11)
 
    # Remote Transmission Request (RTR) bit
    rtr_bit = '0'
    
    # Identifier Extension (IDE) bit
    ide_bit = '0'
 
    # Control bits (R0 and Stuff)
    control_r0_bit = '0'
    # control_stuff_bit = '1'
 
    # Converting Data Length Code (DLC) to 4-bit binary representation
    dlc_bits = bin(dlc)[2:].zfill(4)
 
    # Convert data bytes to binary representation
    # if dlc:
    #     if data[0] != '':
    #         data_bits = ''.join(hex_to_bits(hex_byte, dlc*8) for hex_byte in data)
    #     else:
    #         data_bits = ''
    # else:
    #     data_bits = ''

    data_bits = hex_to_bits(data, dlc * 8)
 
    # Filling missing data bytes with zeros
    # padding_bits = '0' * (8 * (8 - dlc))
    # data_bit_total = data_bits + padding_bits 
    data_bit_total = data_bits
 
    # Calculating CRC-15 checksum and converting to binary representation
    crc_bit = bin(calculate_crc(start_of_frame + can_id_bits + rtr_bit + ide_bit + control_r0_bit +
                                dlc_bits + data_bit_total))[2:].zfill(15)
 
    # CRC delimiter bit
    crc_delimiter = '1'
 
    # Acknowledge (ACK) bit
    ack_bit = '0'
 
    # ACK delimiter bit
    ack_delimiter = '1'
 
    # End of Frame (EOF) bits
    end_of_frame_bits = '1' * 7
 
    # Inter-Frame Spacing bits
    inter_frame_spacing_bits = '1' * 3
 
    # stuffing the bits:
    return len(stuff_bits(start_of_frame + can_id_bits + rtr_bit + 
                          ide_bit + control_r0_bit +  dlc_bits +
                          data_bit_total + crc_bit + 
                          crc_delimiter + ack_bit +ack_delimiter + 
                          end_of_frame_bits + inter_frame_spacing_bits))

def transmission_time(frame_length, bus_rate=500):
    
    """
    Gives the time to transmit a packet given its frame length

    Args:
        frame_length (int): Length of the binary CAN packet frame  
        bus_rate (int): Busrate in Kbps
    Returns:
        float: Time to transmit the packet onto CAN bus in seconds
    """

    return frame_length/(bus_rate * 1000)



