from modules.entities_NLU import EntityTracker
from modules.bow import BoW_encoder
from modules.lstm_net import LSTM_net
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker

import numpy as np
import copy


class InteractiveSession:

    def __init__(self):

        self.et = EntityTracker()
        self.at = ActionTracker(self.et)

        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()

        obs_size = self.emb.dim + self.bow_enc.vocab_size + self.et.num_features
        self.action_templates = self.at.get_action_templates()
        action_size = self.at.action_size
        nb_hidden = 128

        self.net = LSTM_net(obs_size=obs_size, action_size=action_size, nb_hidden=nb_hidden)

        # restore checkpoint
        self.net.restore()
        self.net.reset_state()

    def reset(self):
        self.net.reset_state()
        self.et = EntityTracker()
        self.at = ActionTracker(self.et)

    def interact(self, utterance, intent, slot_values):
        # get input from user
        u = utterance.lower()

        # check if user wants to begin new session
        if u == 'clear' or u == 'reset' or u == 'restart':
            self.reset()
            return "reset successfully"

        # check for entrance and exit command
        elif u == 'exit' or u == 'stop' or u == 'quit' or u == 'q':
            self.reset()
            return "Thank you for using"

        elif u == 'hello' or u == 'hi':
            self.reset()
            return "what can i do for you"

        elif u == 'thank you' or u == 'thanks' or u == 'thank you very much':
            self.reset()
            return 'you are welcome'

        else:

            # encode
            u_ent = self.et.extract_entities(u, intent, slot_values)
            u_ent_features = self.et.context_features()  # 5
            u_emb = self.emb.encode(u)  # 300
            u_bow = self.bow_enc.encode(u)  # 60
            # concat features
            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)

            # get action mask
            action_mask = self.at.action_mask()
            # action_mask = np.ones(self.net.action_size)

            # forward
            prediction = self.net.forward(features, action_mask)
            response = self.action_templates[prediction]
            if prediction == 0:
                slot_values = copy.deepcopy(self.et.entities)
                slot_values.pop('location')
                memory = ', '.join(slot_values.values())
                response = response.replace("memory", memory)
                self.reset()
                print('API CALL execute successfully and begin new session')
            if prediction == 1:
                response = response.replace("location", self.et.entities['location'])
            return response


if __name__ == '__main__':
    # create interactive session
    isess = InteractiveSession()
    # begin interaction
    utterance = "hello"
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'Greeting', {}))

    utterance = 'I want to make an appointment with the fever clinic department of the hospital'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'department': 'fever clinic department'}))

    utterance = 'Is there any hospital available in Gulou'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'location': 'Gulou'}))

    utterance = 'I\'d like Nanjing Gulou Hospital'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'hospital': 'Nanjing Gulou Hospital'}))

    utterance = 'this afternoon please'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'time': 'this afternoon'}))

    utterance = 'I am Zhong Nanshan, 84 years old, 13333333333'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'person_name': 'Zhong Nanshan', 'age': '84 years old', 'phone_number': '13333333333'}))

    utterance = "thanks"
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'thanks', {}))

