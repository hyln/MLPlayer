import numpy as np
import cv2
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])


# 求协方差矩阵的特征值和特征向量
def pca(inputs):
    # 1. 将样本平均归一化
    avg_vector = inputs - np.mean(inputs, axis=0)
    cov = np.dot(avg_vector.T, avg_vector) / inputs.shape[0]
    # 2. 求特征值和特征向量
    # eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # np.save("val.npy", eigenvalues)
    # np.save("vec.npy", eigenvectors)
    eigenvalues = np.load("val.npy")
    eigenvectors = np.load("vec.npy")
    # 3. 将特征值从大到小排序
    sort = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return avg_vector, eigenvalues, eigenvectors


# 根据pca的结果将图片降维
def dim_reduction(inputs, eigenvectors, dim):
    return np.dot(inputs, eigenvectors[:, :dim])


# 根据pca的结果将图片重建
def reconstruct(inputs, avg_vector, eigenvectors, dim):
    res = np.dot(inputs, eigenvectors[:, :dim])
    res = np.dot(res, eigenvectors[:, :dim].T)
    res += avg_vector
    return res


def plot(inputs):
    avg_vector, eigenvalues, eigenvectors = pca(inputs)

    # 画保留信息的比例的趋势线(随着维度的变化)
    plt.figure()
    x = np.arange(1, 501, 1)
    y = [np.sum(eigenvalues[:k]) / np.sum(eigenvalues) for k in range(1, 501)]
    plt.plot(x, y)

    # 画三条虚线
    plt.plot([0, 111], [0.90, 0.90], c="b", linestyle="--")
    plt.plot([111, 111], [0, 0.90], c="b", linestyle="--")
    plt.plot([0, 190], [0.95, 0.95], c="b", linestyle="--")
    plt.plot([190, 190], [0, 0.95], c="b", linestyle="--")
    plt.plot([0, 325], [0.99, 0.99], c="b", linestyle="--")
    plt.plot([325, 325], [0, 0.99], c="b", linestyle="--")

    # 移动坐标轴
    ax = plt.gca()
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    ax.autoscale(tight=True)
    ax.set(**dict(xlabel="dimension number", ylabel="eigenvalue ratio"))
    plt.tight_layout()
    plt.savefig("fig.png", dpi=300)

    # 显示降维后的图片的例子
    for k in [11, 20, 44, 111, 190, 325]:
        res = reconstruct(inputs, avg_vector, eigenvectors, dim=k)
        cv2.imwrite("image_{}.png".format(k), res[0].reshape((112, 92)))


if __name__ == "__main__":
    from min_dist import read_images_to_vector

    images = read_images_to_vector()
    plot(images)
