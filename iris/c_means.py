import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # 加载数据
    iris_data = load_iris()
    data = iris_data.data
    label = iris_data.target

    # 将数据降到二维空间方便可视化
    tsne = TSNE(n_components=2, init="random", random_state=2).fit(data)
    df = pd.DataFrame(tsne.embedding_)
    data1 = df[label == 0]
    data2 = df[label == 1]
    data3 = df[label == 2]

    # 将原始数据点可视化
    plt.plot(data1[0], data1[1], "bo")
    plt.plot(data2[0], data2[1], "r*")
    plt.plot(data3[0], data3[1], "gD")

    # 聚类
    cluster = KMeans(n_clusters=3, random_state=2).fit(data)
    data1_cluster = df[cluster.labels_ == 0]
    data2_cluster = df[cluster.labels_ == 1]
    data3_cluster = df[cluster.labels_ == 2]
    plt.plot(data1_cluster[0], data1_cluster[1], "bo")
    plt.plot(data2_cluster[0], data2_cluster[1], "r*")
    plt.plot(data3_cluster[0], data3_cluster[1], "gD")
    plt.show()

    sum = [0, 0, 0]
    for i in cluster.labels_:
        sum[i] += 1
    print("第一类的数量为: {}, 类中心为: {}".format(sum[0], cluster.cluster_centers_[0]))
    print("第二类的数量为: {}, 类中心为: {}".format(sum[1], cluster.cluster_centers_[1]))
    print("第三类的数量为: {}, 类中心为: {}".format(sum[2], cluster.cluster_centers_[2]))
    print(
        "迭代次数为: {}, 错误数为: {}, 错误率为: {}".format(
            cluster.n_iter_,
            np.sum(cluster.labels_ != label),
            np.sum(cluster.labels_ != label) / len(label),
        )
    )
