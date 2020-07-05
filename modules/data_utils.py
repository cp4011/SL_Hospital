import modules.util as util

'''
    Train

    1. Prepare training examples
        1.1 Format 'utterance \t action_template_id\n'
    2. Prepare dev set
    3. Organize trainset as list of dialogues
'''


class Data:

    def __init__(self, entity_tracker, action_tracker):

        self.action_templates = action_tracker.get_action_templates()
        self.et = entity_tracker
        # prepare data
        self.trainset = self.prepare_data()

    def prepare_data(self):
        # get dialogs from file
        dialogs, dialog_indices = util.read_dialogs(with_indices=True)
        # get utterances
        utterances = util.get_utterances(dialogs)
        # get responses
        responses_id = util.get_responses()

        trainset = []
        for u, r in zip(utterances, responses_id):
            trainset.append((u, int(r)-1))

        return trainset, dialog_indices     # [(utterance_1, action_template_id_1),..]  [{'start':0, 'end':20},...]

