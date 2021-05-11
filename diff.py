import random

import numpy as np


class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))  # 마찬가지로 Q 테이블을 0으로 초기화
        self.eps = 0.9

    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 3)
        else:
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        a_prime = self.select_action(s_prime)  # S'에서 선택할 액션 (실제로 취한 액션이 아님)
        # Q러닝 업데이트 식을 이용
        self.q_table[x, y, a] = self.q_table[x, y, a] + 0.1 * (
                r + np.amax(self.q_table[next_x, next_y, :]) - self.q_table[x, y, a])

    def anneal_eps(self):
        self.eps -= 0.01  # Q러닝에선 epsilon 이 좀더 천천히 줄어 들도록 함.
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        q_lst = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)
