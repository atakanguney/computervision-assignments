from matplotlib import pyplot as plt
import numpy as np
	
#%%
#plt.clf()
plt.figure()
img1 = plt.imread("cmpe-building/left-2.jpg")
plt.imshow(img1)
im1Points = plt.ginput(15)
im1Points = [tuple(reversed(t)) for t in im1Points]

plt.figure()
img2 = plt.imread("cmpe-building/left-1.jpg")
plt.imshow(img2)
im2Points = plt.ginput(15)
im2Points = [tuple(reversed(t)) for t in im2Points]

#%%

im1Points = np.array(im1Points)
im2Points = np.array(im2Points)

def normalize(imPoints):
    avg = imPoints.mean(axis=0)
    imPoints = imPoints - avg
    
    pov = np.linalg.norm(imPoints, axis=1).mean() / np.sqrt(2)
    imPoints = imPoints / pov
    
    M = np.identity(imPoints.shape[1] + 1)
    M[:-1, -1] = -avg
    
    V = np.identity(imPoints.shape[1] + 1)
    np.fill_diagonal(V, 1/pov)
    V[-1, -1] = 1
    
    T = V @ M
    return imPoints, T
    
im1Points_norm, T_1 = normalize(im1Points)
im2Points_norm, T_2 = normalize(im2Points)

#%%
#find homography matrix
def corr_mat(pt1, pt2):
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    
    return np.array([[x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2],
                     [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2]])

P = np.vstack(corr_mat(pt1, pt2) for pt1,pt2 in zip(im1Points_norm, im2Points_norm))

_, _, V = np.linalg.svd(P)
h = V[-1, :]
H_norm = h.reshape((3,3))

H = np.linalg.inv(T_2) @ H_norm @ T_1 

#%%
#image warping
M, N, _ = img1.shape
img_warp = np.zeros(img1.shape, dtype="int")
H_inv = np.linalg.inv(H)


for i in range(M):
    for j in range(N):
        src = H_inv @ np.array([[i], [j], [1]])
        x, y = int(src[0][0] / src[2][0]), int(src[1][0] / src[2][0])
        if(M > x > -1 and N > y > -1):
            img_warp[i, j, :] += img1[x, y,:]

#for i in range(M):
#    for j in range(N):
#        src = H @ np.array([i, j, 1])
#        x, y = int(src[0] / src[2]), int(src[1] / src[2])
#        if(M > x > -1 and N > y > -1 and (img_warp[x, y, :]==0).all()):
#            img_warp[x, y, :] = img1[i, j,:]

plt.figure(); plt.imshow(img_warp)

#%%
M, N, _  = img1.shape
corners = np.array([[0, 0, 1], 
                    [0, N, 1], 
                    [M, 0, 1], 
                    [M, N, 1]])

corners_map = H @ corners.transpose()
corners_map = corners_map[:2,:] / corners_map[2]

max_x, max_y = corners_map.max(axis=1).astype(int)
min_x, min_y = corners_map.min(axis=1).astype(int)
max_x = max(img2.shape[0], max_x)
min_x = min(0, min_x)
max_y = max(img2.shape[1], max_y)
min_y = min(0, min_y)


img_warp = np.zeros((max_x - min_x, max_y - min_y, 3), dtype="int")
for i in range(min_x, max_x):
    for j in range(min_y, max_x):
        src = H_inv @ np.array([[i], [j], [1]])
        x, y = int(src[0][0] / src[2][0]), int(src[1][0] / src[2][0])
        if(M > x > -1 and N > y > -1):
            img_warp[i - min_x -1, j - min_y - 1, :] += img1[x, y,:]
            
#%%
img_warp[-min_x:M-min_x, -min_y:, :] = img2
plt.figure(); plt.imshow(img_warp)
plt.show()
