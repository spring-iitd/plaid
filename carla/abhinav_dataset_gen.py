

import random

class DatasetGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w')
        self.data = []

    def __flush_data(self):
        for d in self.data:
            self.file.write(d)
        self.data = []

    def close(self):
        if self.file:
            self.__flush_data()
            self.file.close()
            print('Dataset saved (overwritten) to <{}>'.format(self.filename))
        else:
            print('WARNING: ISSUE WITH FILE OPENING')

    def add_message(self, timestamp, can_id, data):
        log_line = f"({timestamp:.6f})  vcan0  {can_id:08X}   [8]  {' '.join([f'{byte:02X}' for byte in data])}\n"
        self.data.append(log_line)
        if (len(self.data) > 100000):
            self.__flush_data()


    def add_random_message(self, timestamp):
        can_id = random.randint(0, 0xFFFFFFFF)
        data = [random.randint(0, 255) for _ in range(8)]
        self.add_message(timestamp, can_id, data)


messages = {
    'throttle' : [380, [0xFA, 0x00, 0x36, 0xB0, 0x83, 0x00, 0x40, 0x00]],
    'brake' : [380, [0x32, 0x00, 0x03, 0xe8, 0x00, 0x00, 0x20, 0x00]],
    'left' : [330, [0xFF, 0xFF, 0x1F, 0x40, 0x03, 0x00, 0x40, 0x00]],
    'right' : [330, [0x01, 0x00, 0x1F, 0x40, 0x03, 0x00, 0x40, 0x00]],
}

probabilities = {
    'throttle' : 100,
    'brake' : 50,
    'left' : 20,
    'right' : 40,
    'others' : 50
}
seed = 4
dataset_size = 10000
start_time = 10.0 # seconds
increment = 0.1




if __name__ == "__main__":
    # Example usage
    dg = DatasetGenerator("abhinav_spoofing_dataset.log")

    random.seed(seed)
    timestamp = start_time
    for _ in range(dataset_size):
        message_type = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
        if (message_type == 'others'):
            dg.add_random_message(timestamp)
        else:
            can_id, data = messages[message_type]
            dg.add_message(timestamp, can_id, data)
        timestamp += increment

    dg.close()












