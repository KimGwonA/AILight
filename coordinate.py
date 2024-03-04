import numpy as np
import math

#영상좌표 (x,y)
x = 100
y = 100


#주점과 초점거리는 사실상 고정값이라 최초 한번만 구하면 됨

#주점 (cx,cy)
#주점의 이상치는 해상도의 절반
cx = 960
cy = 540
#초점거리 (fx, fy)
#초점거리는 카메라 켈리브레이션으로 구해야함
fx = 2050
fy = 2050
#각도 p(pan) / tilt(t)
p = 0
t = 0

#영상좌표에 대한 정규 이미지 좌표(u, v) 계산
u = (x - cx) / fx
v = (y - cy) / fy


Xc = np.array([
    [u],
    [v],
    [1]
])

# Xc를 월드 좌표로 변환
# cal = np.array([
#     [math.cos(p), -1 * math.sin(p) * math.sin(t), -1 * math.sin(p) * math.cos(t)],
#     [math.sin(p), math.cos(p) * math.sin(t), math.cos(p) * math.cos(t)],
#     [0, -1 * math.cos(t), math.sin(t)],
# ])
cal = np.array([
    [math.cos(p), -math.sin(p) * math.sin(t), -math.sin(p) * math.cos(t)],
    [math.sin(p), math.cos(p) * math.sin(t), math.cos(p) * math.cos(t)],
    [0, -math.cos(t), math.sin(t)],
])

# Xw = cal * Xc
Xw = np.dot(cal, Xc)

# Xw = (a, b, c)
# a = Xw[0][0]
# b = Xw[0][1]
# c = Xw[0][2]

a = Xw[0][0]
b = Xw[1][0]
c = Xw[2][0]



# Xw의 팬각(tar_p), 틸트각(tar_t)계산
tar_p = -1 * math.atan2(a, b)
tar_t = math.atan2(c, math.sqrt(a*a + b*b))

go_tar_p = tar_p - p
go_tar_t = tar_t - t

# 57 곱하기(라디언으로 나온 값 변경)
go_tar_p = round(go_tar_p, 2) * 57
go_tar_t = round(go_tar_t, 2) * 57

print(go_tar_p, go_tar_t)



