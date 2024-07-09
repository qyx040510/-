import pygame
import sys

def display_recognition_result(result):
    # 初始化pygame
    pygame.init()

    # 设置显示窗口的大小
    screen = pygame.display.set_mode((800, 600))  # 根据你的显示屏调整分辨率
    pygame.display.set_caption("Gait Recognition Result")

    # 加载图片
    stranger_image = pygame.image.load("stranger.png")
    person_images = {
        1: pygame.image.load("person_1.png"),
        2: pygame.image.load("person_2.png"),
        3: pygame.image.load("person_3.png")
        # 添加更多人员ID及其对应的图片
    }

    # 根据步态识别结果显示相应的图片
    if result == 0:
        screen.blit(stranger_image, (0, 0))
    else:
        image = person_images.get(result, stranger_image)
        screen.blit(image, (0, 0))
    pygame.display.flip()

    # 主循环，保持窗口打开
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    sys.exit()
