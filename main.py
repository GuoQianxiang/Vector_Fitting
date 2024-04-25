import Loader.data_loader as Loader

if __name__ == "__main__":

    data, Z0, frq, odc = Loader.load("./Data/data.mat")
    # 打印参数的形状和数值
    print("Z0 shape:", Z0.shape)
    print("frq shape:", frq.shape)
    print("frq values:", frq)
    print("odc:", odc)
