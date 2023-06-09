import os
import time

import nibabel
import numpy as np
import matplotlib.pyplot as plt
import collections
import yaml
from queue import Queue
from pydicom.filereader import read_file
from utils.progress_bar import format_time
from skimage import measure

def draw_hist(inferred_num, data_folder):
    filters = []
    for i in inferred_num:

        path1 = os.path.join(data_folder, 'spine1.0-nii', 'case'+i+'.nii.gz')
        path2 = os.path.join(data_folder, 'spine1.0-nii', 'mask'+i+'.nii.gz')

        img = nibabel.load(path1).get_fdata()
        mask = nibabel.load(path2).get_fdata()
        x = img * mask
        spine = (img * mask).astype(np.int32).ravel()
        filter = [i for i in spine if i > 0]
        print(min(filter), max(filter))
        filters.append(filter)

    print(len(filters[0]))
    print(len(filters[1]))
    print(len(filters[2]))

    plt.hist(filters[0],bins=50, weights=np.zeros_like(filters[0]) + 1. / len(filters[0]), color='r')
    # plt.show()
    plt.hist(filters[1],bins=50, weights=np.zeros_like(filters[1]) + 1. / len(filters[1]), color='g')
    # plt.show()
    plt.hist(filters[2],bins=50, weights=np.zeros_like(filters[2]) + 1. / len(filters[2]), color='b')
    plt.show()


# 'kernel' was applied in (row, col, index)
def multiply_kernel(matrix, row, col, index):
    # matrix[row, col, index] = 255
    if row-1 >= 0:
        matrix[row-1, col, index]=255

    if row+1 < matrix.shape[0] :
        matrix[row+1, col, index] = 255

    if col-1 >= 0:
        matrix[row, col-1, index] = 255

    if col+1 < matrix.shape[1]:
        matrix[row, col+1, index] = 255

    if index-1 >= 0:
        matrix[row, col, index-1] = 255

    if index+1 < matrix.shape[2]:
        matrix[row, col, index+1] = 255

    return matrix

# 反转
def invertImage(img):
    row = img.shape[0]
    col = img.shape[1]
    index = img.shape[2]
    output = np.zeros((row, col, index))
    for i in range(row):
        for j in range(col):
            for k in range(index):
                if img[i, j, k] == 255:
                    output[i, j, k] = 0
                elif img[i, j, k] == 0:
                    output[i, j, k] = 255

    return output

# 膨胀
def dilation(image, kernel):
    row = image.shape[0]
    col = image.shape[1]
    index = image.shape[2]
    output = np.copy(image)
    for i in range(row):
        for j in range(col):
            for k in range(index):
                if image[i, j, k] == 255:
                    output = multiply_kernel(output, i, j, k)
    return output

# 腐蚀
def erosion(img, kernel):
    invertedimg = invertImage(img)
    dilated = dilation(invertedimg, kernel)
    output = invertImage(dilated)
    return output

# 开运算
def opening(img,kernel):
    return dilation(erosion(img,kernel),kernel)
# 闭运算
def closing(img,kernel):
    return erosion(dilation(img,kernel),kernel)

# the shape of kernel can be described like this
kernel = [
    [[0,0,0],[0,1,0],[0,0,0]],
    [[0,1,0],[1,1,1],[0,1,0]],
    [[0,0,0],[0,1,0],[0,0,0]]
]
kernel = np.array(kernel)

# form dst-mask
def form_dst_mask(data_folder, inferred_num):
    for i in inferred_num:
        begin_time = time.time()
        path1 = os.path.join(data_folder, 'spine1.0-nii', 'case' + i + '.nii.gz')
        path2 = os.path.join(data_folder, 'experiments/2D-Sagittal/inferred_results', 'case' + i + '_inferred.nii.gz')

        img = nibabel.load(path1).get_fdata()
        mask = nibabel.load(path2).get_fdata()

        spine = img * mask
        affine = nibabel.load(path1).affine

        # python对象，赋值后是同一地址，可变对象的修改会影响到所有对象，copy()创建副本
        mask_flag_dense = spine.copy()
        mask_flag_sparse = spine.copy()
        # 密质骨
        mask_flag_dense[mask_flag_dense <= 300.0] = 0
        mask_flag_dense[mask_flag_dense != 0] = 1
        # 松质骨
        mask_flag_sparse[mask_flag_sparse > 300.0] = 0
        mask_flag_sparse[mask_flag_sparse != 0] = 1

        # dst_mask_dense = spine * mask_flag_dense
        # dst_mask_dense = opening(dst_mask_dense, kernel)
        # nft_dense = nibabel.Nifti1Image(dst_mask_dense, affine)
        # nibabel.save(nft_dense, os.path.join('./experiments/results', 'dst_mask_dense_' + str(i).zfill(3) + '.nii.gz'))

        dst_mask_sparse = spine * mask_flag_sparse
        dst_mask_sparse = opening(dst_mask_sparse, kernel)
        nft_spare = nibabel.Nifti1Image(dst_mask_sparse, affine)

        nibabel.save(nft_spare, os.path.join(save_folder, 'dst_mask_spare_' + i + '.nii.gz'))
        cur_time = time.time()
        tot_time = cur_time - begin_time
        print('form dst-mask case %s cost time: %s' % (i, format_time(tot_time)))


# 24 min in each case
class HoleFill_2D():
    def __init__(self, grid, affine, i):

        self.grid = grid
        self.affine = affine
        self.n = len(grid)
        self.m = len(grid[0])
        self.s = len(grid[0][0])
        self.idx = i

    def solve(self, save_folder):
        begin_time = time.time()
        for z in range(self.s):
            slice_z = self.grid[:, :, z].copy()
            if slice_z.all():
                return
            que = collections.deque()
            for i in range(self.n):
                if slice_z[i][0] == 0:
                    que.append((i, 0))
                    slice_z[i][0] = 1
                if slice_z[i][self.m - 1] == 0:
                    que.append((i, self.m - 1))
                    slice_z[i][self.m - 1] = 1
            for i in range(self.m - 1):
                if slice_z[0][i] == 0:
                    que.append((0, i))
                    slice_z[0][i] = 1
                if slice_z[self.n - 1][i] == 0:
                    que.append((self.n - 1, i))
                    slice_z[self.n - 1][i] = 1
            
            while que:
                x, y = que.popleft()
                for mx, my in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= mx < self.n and 0 <= my < self.m and slice_z[mx][my] == 0:
                        que.append((mx, my))
                        slice_z[mx][my] = 1
            
            for i in range(self.n):
                for j in range(self.m):
                    if slice_z[i][j] == 1:
                        slice_z[i][j] = 0
                    elif slice_z[i][j] == 0:
                        slice_z[i][j] = 255
            self.grid[:, :, z] = slice_z

        for y in range(self.m):
            slice_y = self.grid[:, y, :].copy()     
            if slice_y.all():
                return
            que = collections.deque()
            for i in range(self.n):
                if slice_y[i][0] == 0:
                    que.append((i, 0))
                    slice_y[i][0] = 1
                if slice_y[i][self.s - 1] == 0:
                    que.append((i, self.s - 1))
                    slice_y[i][self.s - 1] = 1
            for i in range(self.s - 1):
                if slice_y[0][i] == 0:
                    que.append((0, i))
                    slice_y[0][i] = 1
                if slice_y[self.n - 1][i] == 0:
                    que.append((self.n - 1, i))
                    slice_y[self.n - 1][i] = 1

            while que:
                x, y = que.popleft()
                for mx, my in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= mx < self.n and 0 <= my < self.s and slice_y[mx][my] == 0:
                        que.append((mx, my))
                        slice_y[mx][my] = 1

            for i in range(self.n):
                for j in range(self.s):
                    if slice_y[i][j] == 1:
                        slice_y[i][j] = 0
                    elif slice_y[i][j] == 0:
                        slice_y[i][j] = 255
            self.grid[:, y, :] = slice_y

        for x in range(self.n):
            slice_x = self.grid[x, :, :].copy()     
            if slice_x.all():
                return
            que = collections.deque()
            for i in range(self.m):
                if slice_x[i][0] == 0:
                    que.append((i, 0))
                    slice_x[i][0] = 1
                if slice_x[i][self.s - 1] == 0:
                    que.append((i, self.s - 1))
                    slice_x[i][self.s - 1] = 1
            for i in range(self.s - 1):
                if slice_x[0][i] == 0:
                    que.append((0, i))
                    slice_x[0][i] = 1
                if slice_x[self.m - 1][i] == 0:
                    que.append((self.m - 1, i))
                    slice_x[self.m - 1][i] = 1

            while que:
                x, y = que.popleft()
                for mx, my in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= mx < self.m and 0 <= my < self.s and slice_x[mx][my] == 0:
                        que.append((mx, my))
                        slice_x[mx][my] = 1

            for i in range(self.m):
                for j in range(self.s):
                    if slice_x[i][j] == 1:
                        slice_x[i][j] = 0
                    elif slice_x[i][j] == 0:
                        slice_x[i][j] = 255
            self.grid[x, :, :]  = slice_x

        nft = nibabel.Nifti1Image(self.grid, self.affine)
        nibabel.save(nft, os.path.join(save_folder, 'filled_dst_mask_spare_' + self.idx + '.nii.gz'))
        cur_time = time.time()
        tot_time = cur_time - begin_time
        print('form filled mask case %s cost time: %s' % (self.idx, format_time(tot_time)))


def HoleFillEachCase(save_folder, inferred_num):
    for i in inferred_num:
        path = os.path.join(save_folder, 'dst_mask_spare_' + i + '.nii.gz')
        mask = nibabel.load(path).get_fdata()
        affine = nibabel.load(path).affine
        holefill_2d = HoleFill_2D(mask, affine, i)
        holefill_2d.solve(save_folder)



# DFS - 超出最大递归深度,栈
class DFS_InstanceSearch():

    def __init__(self, grid, affine, i):

        self.grid = grid
        self.affine = affine
        self.idx = i
        self.n = len(grid)
        self.m = len(grid[0])
        self.s = len(grid[0][0])

    # 深度优先遍历与i，j相邻的所有1
    def dfs(self, grid, id, i, j, k):
        # 像素值置为标签ID
        self.grid[i][j][k] = id

        # 周围三维空间中6个方向遍历
        if i - 1 >= 0 and grid[i - 1][j][k] == 255:
            self.dfs(self.grid, id, i - 1, j, k)
        if i + 1 < self.n and grid[i + 1][j][k] == 255:
            self.dfs(self.grid, id, i + 1, j, k)
        if j - 1 >= 0 and grid[i][j - 1][k] == 255:
            self.dfs(self.grid, id, i, j - 1, k)
        if j + 1 < self.m and grid[i][j + 1][k] == 255:
            self.dfs(self.grid, id, i, j + 1, k)
        if k - 1 >= 0 and grid[i][j][k - 1] == 255:
            self.dfs(self.grid, id, i, j, k - 1)
        if k + 1 < self.s and grid[i][j][k + 1] == 255:
            self.dfs(self.grid, id, i, j, k + 1)

    def solve(self, save_folder):
        ids = iter(range(1, 1000))
        num = 0
        # 遍历图像矩阵
        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    # 遍历到1的情况
                    if self.grid[i][j][k] == 255:
                        # 取标签
                        print(i, j, k)
                        id = next(ids)
                        print(id, num)
                        num += 1
                        # 搜索赋id
                        self.dfs(self.grid, id, i, j, k)

        nft = nibabel.Nifti1Image(self.grid, self.affine)
        nibabel.save(nft, os.path.join(save_folder, 'instance_mask_'+ str(self.idx).zfill(3) +'.nii.gz'))


class BFS_InstanceSearch():

    def __init__(self, grid, affine, i):

        self.grid = grid
        self.affine = affine
        self.idx = i
        self.n = len(grid)
        self.m = len(grid[0])
        self.s = len(grid[0][0])
        self.island = 0

    def solve(self, save_folder):
        begin_time = time.time()
        ids = iter(range(1, 1000))
        # 遍历像素矩阵
        ''' 单例搜索连通域用时 1m30s '''
        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    # 遇到'255'将这个'255'及与其相邻的'255'都置为标签值
                    if self.grid[i][j][k] == 255:
                        # print('上一个连通域像素点数量:', self.island)
                        self.island = 0
                        id = next(ids)
                        self.grid[i][j][k] = id
                        # print('连通域label:', id)
                        # 记录后续bfs的坐标
                        q = Queue()
                        q.put([i, j, k])
                        # bfs
                        while not q.empty():
                            temp = q.get()
                            row = temp[0]
                            col = temp[1]
                            depth = temp[2]
                            # 6个方向依次检查：不越界且不为0
                            if row - 1 >= 0 and self.grid[row - 1][col][depth] == 255:
                                q.put([row - 1, col, depth])
                                self.grid[row - 1][col][depth] = id
                            if row + 1 < self.n and self.grid[row + 1][col][depth] == 255:
                                q.put([row + 1, col, depth])
                                self.grid[row + 1][col][depth] = id
                            if col - 1 >= 0 and self.grid[row][col - 1][depth] == 255:
                                q.put([row, col - 1, depth])
                                self.grid[row][col - 1][depth] = id
                            if col + 1 < self.m and self.grid[row][col + 1][depth] == 255:
                                q.put([row, col + 1, depth])
                                self.grid[row][col + 1][depth] = id
                            if depth - 1 >= 0 and self.grid[row][col][depth - 1] == 255:
                                q.put([row, col, depth - 1])
                                self.grid[row][col][depth - 1] = id
                            if depth + 1 < self.s and self.grid[row][col][depth + 1] == 255:
                                q.put([row, col, depth + 1])
                                self.grid[row][col][depth + 1] = id

                            self.island += 1

        elapsed_time = time.time() - begin_time
        print('search connected domain in case %s cost time: %s' % (self.idx, format_time(elapsed_time)))

        # 取连通域像素点总数第2~第18的区域(第1是背景像素'0')
        ''' 单例像素值排序用时 4m15s '''
        grid_filter = list(set(self.grid.ravel()))
        # 椎体不足17例时，补齐标签(虽然value为0)
        while len(grid_filter) < 18:
            grid_filter.append(float(len(grid_filter)))

        intensity_dict = dict()

        for i in range(len(grid_filter)):
            intensity_dict[grid_filter[i]] = 0

        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    intensity = self.grid[i][j][k]
                    intensity_dict[intensity] = intensity_dict[intensity] + 1
        print('像素值:像素点数量 - \n', intensity_dict)
        sorted_intensity = sorted(intensity_dict.items(), key=lambda s: s[1], reverse=True)
        print('像素值:像素点数量 (按数量逆序) - \n', sorted_intensity)
        visual_intensity = []
        for i in range(1, 18):
            visual_intensity.append(sorted_intensity[i][0])

        ranked_intensity = []
        for k in range(self.s):
            for i in range(self.n):
                for j in range(self.m):
                    if self.grid[i][j][k] not in visual_intensity:
                        self.grid[i][j][k] = 0
                    else:
                        # ①对椎体编序号
                        # 按连通域体积大小排序对像素值设定为1-17，但必须保证椎体体积有序递减
                        # idx = visual_intensity.index(intensity)
                        # self.grid[i][j][k] = idx + 1

                        # ②对椎体编序号
                        # 在z轴(self.s)方向上自顶向下(T12可能会缺失)，将新的可见像素按序保存：保证了'像素值顺序'与'椎体顺序'一致
                        if self.grid[i][j][k] not in ranked_intensity:
                            ranked_intensity.append(self.grid[i][j][k])
                        pass
        print('z轴(self.s)方向上自顶向下的像素值顺序,即T1~T12-S1~S5:\n', list(reversed(ranked_intensity)))

        # ②对椎体编序号
        # 对mask中像素值进行有序赋值1-17
        ''' 单例赋值用时 4m10s '''
        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    if self.grid[i][j][k] in list(reversed(ranked_intensity)):
                        idx = list(reversed(ranked_intensity)).index(self.grid[i][j][k])
                        self.grid[i][j][k] = idx +1.5

        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    if self.grid[i][j][k] != 0:
                        self.grid[i][j][k] -= 0.5

        nft = nibabel.Nifti1Image(self.grid, self.affine)
        nibabel.save(nft, os.path.join(save_folder, 'instance_mask_' + self.idx + '.nii.gz'))

        ''' 单例总用时 9m50s '''
        elapsed_time = time.time() - begin_time
        print('form instance case %s cost time: %s \n' % (self.idx, format_time(elapsed_time)))


class Skimage_InstanceSearch():

    def __init__(self, grid, affine, i):

        self.grid = grid
        self.affine = affine
        self.idx = i
        self.n = len(grid)
        self.m = len(grid[0])
        self.s = len(grid[0][0])
        self.island = 0

    def solve(self, save_folder):
        begin_time = time.time()
        ''' 单例搜索连通域用时 5s '''
        self.grid = measure.label(self.grid) # int64
        elapsed_time = time.time() - begin_time
        print('search connected domain in case %s cost time: %s' % (self.idx, format_time(elapsed_time)))
        self.grid = self.grid.astype(float) # cast to 'float64'
        # 取连通域像素点总数第2~第18的区域(第1是背景像素'0')
        ''' 单例像素值排序用时 4m15s '''
        grid_filter = list(set(self.grid.ravel()))
        # 椎体不足17例时，补齐标签(虽然value为0)
        while len(grid_filter) < 18:
            grid_filter.append(float(len(grid_filter)))

        intensity_dict = dict()

        for i in range(len(grid_filter)):
            intensity_dict[grid_filter[i]] = 0

        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    intensity = self.grid[i][j][k]
                    intensity_dict[intensity] = intensity_dict[intensity] + 1
        print('像素值:像素点数量 - \n', intensity_dict)
        sorted_intensity = sorted(intensity_dict.items(), key=lambda s: s[1], reverse=True)
        print('像素值:像素点数量 (按数量逆序) - \n', sorted_intensity)
        visual_intensity = []
        for i in range(1, 18):
            visual_intensity.append(sorted_intensity[i][0])

        ranked_intensity = []
        for k in range(self.s):
            for i in range(self.n):
                for j in range(self.m):
                    if self.grid[i][j][k] not in visual_intensity:
                        self.grid[i][j][k] = 0
                    else:
                        # ①对椎体编序号
                        # 按连通域体积大小排序对像素值设定为1-17，但必须保证椎体体积有序递减
                        # idx = visual_intensity.index(intensity)
                        # self.grid[i][j][k] = idx + 1

                        # ②对椎体编序号
                        # 在z轴(self.s)方向上自顶向下(T12可能会缺失)，将新的可见像素按序保存：保证了'像素值顺序'与'椎体顺序'一致
                        if self.grid[i][j][k] not in ranked_intensity:
                            ranked_intensity.append(self.grid[i][j][k])
                        pass
        print('z轴(self.s)方向上自顶向下的像素值顺序,即T1~T12-S1~S5:\n', list(reversed(ranked_intensity)))
        # ②对椎体编序号
        # 对mask中像素值进行有序赋值1-17
        ''' 单例赋值用时 4m10s '''
        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    if self.grid[i][j][k] in list(reversed(ranked_intensity)):
                        idx = list(reversed(ranked_intensity)).index(self.grid[i][j][k])
                        self.grid[i][j][k] = idx +1.5

        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.s):
                    if self.grid[i][j][k] != 0:
                        self.grid[i][j][k] -= 0.5

        nft = nibabel.Nifti1Image(self.grid, self.affine)
        nibabel.save(nft, os.path.join(save_folder, 'instance_mask_' + self.idx + '.nii.gz'))

        ''' 单例总用时 9m50s '''
        elapsed_time = time.time() - begin_time
        print('form instance case %s cost time: %s \n' % (self.idx, format_time(elapsed_time)))

def save_instance(save_folder, inferred_num):
    for i in inferred_num:
        path = os.path.join(save_folder, 'dst_mask_spare_' + i + '.nii.gz')

        # global mask
        mask = nibabel.load(path).get_fdata()
        affine = nibabel.load(path).affine

        # 递归
        # dfs_instance_search = DFS_InstanceSearch(mask, affine, i)
        # dfs_instance_search.solve()

        # Queue
        # bfs_instance_search = BFS_InstanceSearch(mask, affine, i)
        # bfs_instance_search.solve(save_folder)

        # 调用skimage
        skimage_instance_search = Skimage_InstanceSearch(mask, affine, i)
        skimage_instance_search.solve(save_folder)


class Dwon_Dir():

    def __init__(self, path):
        self.path = path

    def solve(self):
        lsdir = os.listdir(self.path)

        for i in lsdir:
            if os.path.isdir(os.path.join(self.path, i)):
                self.path = os.path.join(self.path, i)
                self.solve()
            else:
                return self.path


def cal_Hu_value(data_folder, inferred_num):
    for i in inferred_num:
        begin_time = time.time()

        # path1: nii影像文件, path2: 对应nii松质骨标签文件, path3: 对应dcm原文件夹

        path1 = os.path.join(data_folder, 'spine1.0-nii', 'case' + i + '.nii.gz')
        path2 = os.path.join(data_folder, 'experiments/2D-Sagittal/instance_results', 'instance_mask_' + i + '.nii.gz')
        path3 = '/data/P000419494/1.0'

        # 读取dcm tag
        # reference:
        # https://blog.csdn.net/Angle_Cal/article/details/78594839
        # https://blog.csdn.net/dr_yingli/article/details/107196091
        down_dir = Dwon_Dir(path3)
        lastpath = down_dir.solve()
        # dcm_path = os.path.join(path3, 'DICOMDIR')
        dcm_path = os.path.join(lastpath, os.listdir(lastpath)[0])
        df = read_file(dcm_path)
        intercept = float(df.data_element('RescaleIntercept').value)
        slope = float(df.data_element('RescaleSlope').value)

        ## 从img和目标椎体mask求平均Hu
        img = nibabel.load(path1).get_fdata()
        mask = nibabel.load(path2).get_fdata()
        affine = nibabel.load(path1).affine

        # T1~T12椎体mask
        avg_Hu = []
        for idx in range(1, 12):
            dst_mask = mask.copy()
            dst_mask[dst_mask != idx] = 0
            dst_mask[dst_mask == idx] = 1

            dst_img = img * dst_mask

            # Hu =  pixel_val * slope + intercept
            # reference: https://blog.csdn.net/sinolover/article/details/119903209
            count_Hu = dst_mask.sum()
            # ①公式计算
            # sum_Hu = dst_img.sum() * slope + intercept * count_Hu
            # ②非公式计算
            sum_Hu = dst_img.sum()
            avg_Hu.append(round(sum_Hu / count_Hu, 2))
            
            if idx == 6 :
                nft = nibabel.Nifti1Image(dst_mask, affine)
                nibabel.save(nft, os.path.join(data_folder, 'experiments/2D-Sagittal/instance_results', '6th_instance_mask_' + i + '.nii.gz'))

        print('case'+str(i).zfill(3)+'.nii.gz 目标椎体 平均Hu值为:')
        mesg = ''
        for s in range(1, 12):
            mesg = mesg+'T'+str(s)+':'+str(avg_Hu[s-1])+' '
        print(mesg)
        cur_time = time.time()
        tot_time = cur_time - begin_time
        print('cal instance Hu %s cost time: %s \n' % (str(i).zfill(3), format_time(tot_time)))


if __name__ == '__main__':

    with open('experiments/template/config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
    inferred_name_list = [l.strip() for l in open(
        os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        config['dataset']['infer_list']
        )).readlines()]

    inferred_num = []
    for i in inferred_name_list:
        inferred_num.append(i.split('.')[0][-3:])
    
    data_folder = '/data'
    save_folder = os.path.join(os.path.join(data_folder, 'experiments/2D-Sagittal/instance_results'))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    form_dst_mask(data_folder, inferred_num)
    # HoleFillEachCase(save_folder, inferred_num)
    save_instance(save_folder, inferred_num)
    cal_Hu_value(data_folder, inferred_num)