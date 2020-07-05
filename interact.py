from modules.entities import EntityTracker
from modules.bow import BoW_encoder
from modules.lstm_net import LSTM_net
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
import numpy as np
import copy


class InteractiveSession:

    def __init__(self):

        et = EntityTracker()
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()
        at = ActionTracker(et)

        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
        self.action_templates = at.get_action_templates()
        action_size = at.action_size
        nb_hidden = 128

        self.net = LSTM_net(obs_size=obs_size, action_size=action_size, nb_hidden=nb_hidden)

        # restore checkpoint
        self.net.restore()

    def interact(self):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        # begin interaction loop
        while True:

            # get input from user
            u = input('User: ')

            # check if user wants to begin new session
            if u == 'clear' or u == 'reset' or u == 'restart':
                self.net.reset_state()
                et = EntityTracker()
                at = ActionTracker(et)
                print('Bot: Reset successfully')

            # check for entrance and exit command
            elif u == 'exit' or u == 'stop' or u == 'quit' or u == 'q':
                print("Bot: Thank you for using")
                break

            elif u == 'hello' or u == 'hi':
                print("Bot: Hello, what can i do for you")

            elif u == 'thank you' or u == 'thanks' or u == 'thank you very much':
                print('Bot: You are welcome')
                break

            else:
                if not u:
                    continue

                u = u.lower()

                # encode
                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()  # 5

                # print(et.entities)
                # print(et.ctxt_features)

                u_emb = self.emb.encode(u)              # 300
                u_bow = self.bow_enc.encode(u)          # 60
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                # print(features.shape)
                # get action mask
                action_mask = at.action_mask()
                # action_mask = np.ones(self.net.action_size)

                # print("action_mask: ", action_mask)

                # forward
                prediction = self.net.forward(features, action_mask)
                response = self.action_templates[prediction]
                if prediction == 0:
                    slot_values = copy.deepcopy(et.entities)
                    slot_values.pop('<location>')
                    memory = []
                    count = 0
                    for k, v in slot_values.items():
                        memory.append('='.join([k, v]))
                        count += 1
                        if count == 2:
                            memory.append('\n')

                    response = response.replace("memory", ', '.join(memory))

                    # memory = ', '.join(slot_values.values())
                    # response = response.replace("memory", memory)
                    self.net.reset_state()
                    et = EntityTracker()
                    at = ActionTracker(et)
                    # print('Execute successfully and begin new session')
                if prediction == 1:
                    response = response.replace("location", '<location>=' + et.entities['<location>'])
                print('Bot: ', response)


if __name__ == '__main__':
    # create interactive session
    isess = InteractiveSession()
    # begin interaction
    isess.interact()
