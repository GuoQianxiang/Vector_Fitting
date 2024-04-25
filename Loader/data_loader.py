import scipy.io

def load(filename):
    # 加载MATLAB数据文件
    data = scipy.io.loadmat(filename)

    # 访问参数
    Z0 = data['Z0']  # 3*3*43的复数矩阵

    frq = data['frq'].flatten()  # 1*43的double类型矩阵，转换为一维数组

    odc = data['odc'].flatten()  # 整数

    return data, Z0, frq, odc[0]