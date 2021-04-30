import pygame              # ❶ 파이게임 모듈 임포트하기
import time

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
BLOCK_SIZE = 20

RED = 255, 0, 0        # 적색:   적 255, 녹   0, 청   0
GREEN = 0, 255, 0      # 녹색:   적   0, 녹 255, 청   0
BLUE = 0, 0, 255       # 청색:   적   0, 녹   0, 청 255
PURPLE = 127, 0, 127   # 보라색: 적 127, 녹   0, 청 127
BLACK = 0, 0, 0        # 검은색: 적   0, 녹   0, 청   0
GRAY = 127, 127, 127   # 회색:   적 127, 녹 127, 청 127
WHITE = 255, 255, 255  # 하얀색: 적 255, 녹 255, 청 255


def draw_background(screen):
    """게임의 배경을 그린다."""
    background = pygame.Rect((0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, WHITE, background)

def draw_block(screen, color, position):
    """position 위치에 color 색깔의 블록을 그린다."""
    block = pygame.Rect((position[1] * BLOCK_SIZE, position[0] * BLOCK_SIZE),
                        (BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, color, block)

pygame.init()              # ❸ 파이게임을 사용하기 전에 초기화한다.

# ❹ 지정한 크기의 게임 화면 창을 연다.
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

block_position = [0, 0]  # ❶ 블록의 위치 (y, x)

# 종료 이벤트가 발생할 때까지 게임을 계속 진행한다


from datetime import datetime
from datetime import timedelta

# (...) 색 정의, 배경 그리기 함수, 블록 그리기 함수, 파이게임 초기화 코드 생략

DIRECTION_ON_KEY = {
    pygame.K_UP: 'north',
    pygame.K_DOWN: 'south',
    pygame.K_LEFT: 'west',
    pygame.K_RIGHT: 'east',
}

block_direction = 'east'  # ❷ 블록의 방향
block_position = [0, 0]
last_moved_time = datetime.now()
while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:       # 입력된 키가 위쪽 화살표 키인 경우
                block_position[0] -= 1         # 블록의 y 좌표를 1 뺀다
            elif event.key == pygame.K_DOWN:   # 입력된 키가 아래쪽 화살표 키인 경우
                block_position[0] += 1         # 블록의 y 좌표를 1 더한다
            elif event.key == pygame.K_LEFT:   # 입력된 키가 왼쪽 화살표 키인 경우
                block_position[1] -= 1         # 블록의 x 좌표를 1 뺀다
            elif event.key == pygame.K_RIGHT:  # 입력된 키가 왼쪽 화살표 키인 경우
                block_position[1] += 1         # 블록의 x 좌표를 1 더한다
    draw_background(screen)
    draw_block(screen, GREEN, block_position)
    pygame.display.update()

# while True:
#     events = pygame.event.get()
#     for event in events:
#         if event.type == pygame.QUIT:
#             exit()
#         if event.type == pygame.KEYDOWN:
#             # ❸ 입력된 키가 화살표 키면,
#             if event.key in DIRECTION_ON_KEY:
#                 # ❹ 블록의 방향을 화살표 키에 맞게 바꾼다
#                 block_direction = DIRECTION_ON_KEY[event.key]
#
#     if timedelta(seconds=1) <= datetime.now() - last_moved_time:
#         if block_direction == 'north':    # ❺ 1초가 지날 때마다
#             block_position[0] -= 1        # 블록의 방향에 따라
#         elif block_direction == 'south':  # 블록의 위치를 변경한다
#             block_position[0] += 1
#         elif block_direction == 'west':
#             block_position[1] -= 1
#         elif block_direction == 'east':
#             block_position[1] += 1
#         last_moved_time = datetime.now()
#
#     draw_background(screen)
#     draw_block(screen, GREEN, block_position)
#     pygame.display.update()