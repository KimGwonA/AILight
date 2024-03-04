def overlap(rect1):
    '''
    두개의 사각형이 겹치는지 확인하는 함수
    :param rect1: detect된 사각형
    :return: overlap되면 True 아니면 False
    '''
    # xywh to xyxy
    rect1[2] = rect1[0] + rect1[2]
    rect1[3] = rect1[1] + rect1[3]

    #ROI
    # x1 = 250
    # y1 = 350
    # x2 = 400
    # y2 = 500

    x1 = 50
    y1 = 200
    x2 = 110
    y2 = 130


    # print("Checking.....")

    return not(rect1[2] < x1 or
               rect1[0] > x2 or
               rect1[1] > y2 or
               rect1[3] < y1)