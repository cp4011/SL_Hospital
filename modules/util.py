def read_dialogs(with_indices=False):
    with open('data/utterance.txt') as f:
        dialogs = [row.lower() for row in f.read().split('\n')]
        # organize dialogs -> dialog_indices
        prev_idx = -1
        n = 1
        dialog_indices = []     # 存入字典{一段对话 从第几行开始，从第几行结束}
        updated_dialogs = []
        for i, dialog in enumerate(dialogs):
            if not dialogs[i]:       # 处理data中 空行（表示重新开始一段对话）
                dialog_indices.append({
                    'start': prev_idx + 1,
                    'end': i - n + 1
                })
                prev_idx = i - n
                n += 1                  # 到目前i行位置，前面一共有n行空格
            else:
                updated_dialogs.append(dialog)

        if with_indices:
            return updated_dialogs, dialog_indices[:-1]

        return updated_dialogs          # [[utterance_1],...]    # 其实这里的dialog是utterance


def get_utterances(dialogs=[]):     # 返回用户的utterance
    utterance = dialogs if len(dialogs) else read_dialogs()
    return utterance


def get_responses():      # 返回Bot的responses
    with open('data/response_id.txt') as f:
        response_id = ([row.split() for row in f.read().split('\n')])  # '\t'前后为用户和bot的对话
    response_id_list = []
    for row in response_id:
        response_id_list.extend(row)
    return response_id_list


def read_content():     # 将用户的各句utterance拼接起来（用作BOW词库）
    return ' '.join(get_utterances())


am_dict = {
            '00000': [3, 4, 5, 6, 7],
            '00001': [3, 4, 5, 6],
            '00010': [3, 4, 5, 7],
            '00011': [3, 4, 5],
            '00100': [3, 4, 6, 7],
            '00101': [3, 4, 6],
            '00110': [3, 4, 7],
            '00111': [3, 4],
            '01000': [2],
            '01001': [2],
            '01010': [2],
            '01011': [2],
            '01100': [2],
            '01101': [2],
            '01110': [2],
            '01111': [2],
            '10000': [5, 6, 7],
            '10001': [5, 6],
            '10010': [5, 7],
            '10011': [5],
            '10100': [6, 7],
            '10101': [6],
            '10110': [7],
            '10111': [1],
            '11000': [5, 6, 7],
            '11001': [5, 6],
            '11010': [5, 7],
            '11011': [5],
            '11100': [6, 7],
            '11101': [6],
            '11110': [7],
            '11111': [1],
        }
