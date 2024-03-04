import numpy as np
import math

# 영상 좌표 (x, y)
x = 500
y = 500

# 주점 (cx, cy)
cx = 960
cy = 540

# 초점거리 (fx, fy)
fx = 2050
fy = 2050

# 각도 p(pan) / tilt(t)
p = 0
t = 0

# 영상 좌표에 대한 정규 이미지 좌표(u, v) 계산
u = (x - cx) / fx
v = (y - cy) / fy

Xc = np.array([
    [u],
    [v],
    [1]
])

# Xc를 월드 좌표로 변환
cal = np.array([
    [math.cos(t) * math.cos(p), -math.sin(p), -math.sin(t) * math.cos(p)],
    [math.cos(t) * math.sin(p), math.cos(p), -math.sin(t) * math.sin(p)],
    [math.sin(t), 0, math.cos(t)]
])

Xw = np.dot(cal, Xc)

# Xw = (a, b, c)
a = Xw[0][0]
b = Xw[1][0]
c = Xw[2][0]

# Xw의 팬각(tar_p), 틸트각(tar_t) 계산
tar_p = math.atan2(b, a)
tar_t = math.atan2(c, math.sqrt(a * a + b * b))

print("tar_p:", tar_p)
print("tar_t:", tar_t)