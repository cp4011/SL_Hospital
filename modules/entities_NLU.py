import numpy as np


class EntityTracker:

    def __init__(self):
        self.entities = {
            'hospital': None,
            'location': None,
            'department': None,
            'time': None,
            'information': None,
        }
        self.num_features = 5  # tracking 5 entities

    def extract_entities(self, uttrance, intent, slot_values):
        for slot, value in slot_values.items():
            if slot in ['person_name', 'age', 'phone_number']:
                self.entities['information'] = ', '.join(slot_values.values())
                break
            if slot in self.entities:
                self.entities[slot] = value
        return uttrance

    def context_features(self):
        keys = ['hospital', 'location', 'department', 'time', 'information']
        self.ctxt_features = np.array([bool(self.entities[key]) for key in keys], dtype=np.float32)
        return self.ctxt_features
