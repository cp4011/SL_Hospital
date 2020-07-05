import numpy as np


from modules.util import am_dict

'''
    Action Templates
    1. 'api_call register_hospital_department_time_name_age_gender'
    2. 'api_call nearby_hospital_location'
    3. 'ask_hospital'
    4. 'ask_location'
    5. 'ask_department'
    6. 'ask_time'
    7. 'ask_information'
    
    1. 'API_CALL_Register_Hospital(memory)'
    2. 'API_CALL_Nearby_Hospital(location)'
    3. 'Ok which hospital do you want to make an appointment with'
    4. 'May I ask where you are'
    5. 'Ok which department do you want to make an appointment with'
    6. 'Ok when do you want to make an appointment'
    7. 'Can you provide me your personal information (name, age and phone number)'

    
    [1] : hospital
    [2] : location
    [3] : department
    [4] : time
    [5] : information
    
'''
action_template = ['API_CALL_Register_Hospital(memory)', 'API_CALL_Nearby_Hospital(location)', 'Ok which hospital do you want to make an appointment with', 'May I ask where you are', 'Ok which department do you want to make an appointment with', 'Ok when do you want to make an appointment', 'Can you provide me your personal information (name, age and phone number)']


class ActionTracker:

    def __init__(self, ent_tracker):
        # maintain an instance of EntityTracker
        self.et = ent_tracker
        # get a list of action templates
        self.action_templates = self.get_action_templates()
        self.action_size = len(self.action_templates)
        # action mask
        self.am = np.zeros([self.action_size], dtype=np.float32)
        # action mask lookup, built on intuition
        self.am_dict = am_dict

    def action_mask(self):
        self.am = np.zeros([self.action_size], dtype=np.float32)
        ctxt_f = ''.join([str(flag) for flag in self.et.context_features().astype(np.int32)])   # 如'11110'

        accumulated_slot_values = {}
        for k, v in self.et.entities.items():
            if v:
                accumulated_slot_values.update({k: v})
        # print("accumulated slot_values: ", accumulated_slot_values)
        # print("ctxt_f: ", ctxt_f)

        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index - 1] = 1.
            return self.am

        return construct_mask(ctxt_f)       # 返回当前状态的上下文特征 可以执行的action templates的id位置置为1

    def get_action_templates(self):
        return action_template
