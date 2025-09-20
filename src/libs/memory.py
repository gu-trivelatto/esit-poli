import pickle
from abc import ABC


class Memory(ABC):
    def __init__(self):
        self.debug_log_path = 'metadata/debug.log'

    def save_history(self, history):
        with open("metadata/chat_history.pkl", "wb") as f:
            pickle.dump(history, f)
    
    def save_debug(self, debug_string):
        print(debug_string)
        with open(self.debug_log_path, 'a') as f:
            f.write(f'{str(debug_string)}\n')
            
    def get_debug_log_path(self):
        return self.debug_log_path
    
    def save_chat_status(self, status):
        with open('metadata/status.log', 'w') as f:
            f.write(status)