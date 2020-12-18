import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

# np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # 픽셀 수
HEIGHT = 4  # 그리드월드 세로
WIDTH = 12  # 그리드월드 가로


class SarsaEnv(tk.Tk):
    def __init__(self):
        super(SarsaEnv, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('SARSA')
        # self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~1200 까지 세로선
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1) # (x0, y0), (x1, y1)을 잇는 선 생성
        for r in range(0, HEIGHT * UNIT, UNIT):  # # 0~400 까지 가로선
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.agent = canvas.create_image(50, 350, image=self.shapes[0])

        self.cliff1 = canvas.create_image(150, 350, image=self.shapes[1])
        self.cliff2 = canvas.create_image(250, 350, image=self.shapes[1])
        self.cliff3 = canvas.create_image(350, 350, image=self.shapes[1])
        self.cliff4 = canvas.create_image(450, 350, image=self.shapes[1])
        self.cliff5 = canvas.create_image(550, 350, image=self.shapes[1])
        self.cliff6 = canvas.create_image(650, 350, image=self.shapes[1])
        self.cliff7 = canvas.create_image(750, 350, image=self.shapes[1])
        self.cliff8 = canvas.create_image(850, 350, image=self.shapes[1])
        self.cliff9 = canvas.create_image(950, 350, image=self.shapes[1])
        self.cliff10 = canvas.create_image(1050, 350, image=self.shapes[1])

        self.goal = canvas.create_image(1150, 350, image=self.shapes[2])

        canvas.pack()

        return canvas

    def load_images(self):
        agent = PhotoImage(
            Image.open('./img/walle.png').resize((100, 100))) # 65, 65
        cliff = PhotoImage(
            Image.open("./img/thunder.png").resize((100, 100))) # 65, 65
        goal = PhotoImage(
            Image.open("./img/boots.png").resize((100, 100))) # 65, 65

        return agent, cliff, goal

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 9
        else:
            origin_x, origin_y = 42, 58

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(WIDTH):
            for j in range(HEIGHT):
                for action in range(0, 4):
                    state = [i, j]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(j, i, round(temp, 3), action)

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    # def state_to_coords(self, state): # 어디다 씀?
    #     x = int(state[0] * 100 + 50)
    #     y = int(state[1] * 100 + 50)
    #     return [x, y]

    def reset(self):
        self.update()
        # time.sleep(0.5)
        x, y = self.canvas.coords(self.agent)
        self.canvas.move(self.agent, 50 - x, 350 - y) # 초기화하는 위치
        self.render()
        return self.coords_to_state(self.canvas.coords(self.agent))

    def step(self, action):
        state = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # 상
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 하
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 좌
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # 우
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # 에이전트 이동
        self.canvas.move(self.agent, base_action[0], base_action[1])
        # 에이전트를 가장 상위로 배치
        self.canvas.tag_raise(self.agent)
        next_state = self.canvas.coords(self.agent)

        # 보상 함수
        if next_state == self.canvas.coords(self.goal):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.cliff1),
                            self.canvas.coords(self.cliff2),
                            self.canvas.coords(self.cliff3),
                            self.canvas.coords(self.cliff4),
                            self.canvas.coords(self.cliff5),
                            self.canvas.coords(self.cliff6),
                            self.canvas.coords(self.cliff7),
                            self.canvas.coords(self.cliff8),
                            self.canvas.coords(self.cliff9),
                            self.canvas.coords(self.cliff10)]:
            reward = -300
            done = True
        else:
            reward = -1
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        # time.sleep(0.03)
        self.update()