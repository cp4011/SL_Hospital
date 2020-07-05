import numpy as np


class EntityTracker:

    def __init__(self):
        self.entities = {
            '<hospital>': None,
            '<location>': None,
            '<department>': None,
            '<time>': None,
            '<information>': None,
        }
        self.num_features = 5       # tracking 5 entities

        # constants
        self.hospitals = ['nanjing gulou hospital', 'nanjing university hospital', 'nanjing children\'s hospital']
        self.locations = ['in gulou', 'zhujiang road']
        self.departments = ['fever clinic department', 'infectious diseases department', 'respiratory department']
        self.times = ['this afternoon', 'tomorrow', 'the day after tomorrow']
        self.informations = ['zhong nanshan, 84 years old, 13333333333', 'li lanjuan, 73 years old, 15555555555', 'li wenliang, 35 years old, 18888888888', 'chen peng, 23 years old, 18888888888']

    def extract_entities(self, utterance):
        for item in self.hospitals:
            if item in utterance:
                self.entities['<hospital>'] = item
                break
        for item in self.locations:
            if item in utterance:
                self.entities['<location>'] = item
                break
        for item in self.departments:
            if item in utterance:
                self.entities['<department>'] = item
                break
        for item in self.times:
            if item in utterance:
                self.entities['<time>'] = item
                break
        for item in self.informations:
            if item in utterance:
                self.entities['<information>'] = item
                break
        return utterance

    def context_features(self):
        keys = ['<hospital>', '<location>', '<department>', '<time>', '<information>']
        self.ctxt_features = np.array([bool(self.entities[key]) for key in keys], dtype=np.float32)  # 如上下文特征为：01000
        return self.ctxt_features
