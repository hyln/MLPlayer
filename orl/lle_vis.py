import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots


plt.style.use(["science"])


def lle_vis():
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=5)

    X = read_images_to_vector()
    X_reduced = lle.fit_transform(X)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=10)

    plt.plot(
        X_reduced[0][0],
        X_reduced[0][1],
        marker="o",
        markerfacecolor="red",
        markeredgecolor="black",
    )
    plt.plot(
        X_reduced[1][0],
        X_reduced[1][1],
        marker="o",
        markerfacecolor="red",
        markeredgecolor="black",
    )
    plt.plot(
        X_reduced[100][0],
        X_reduced[100][1],
        marker="s",
        markerfacecolor="green",
        markeredgecolor="black",
    )
    plt.plot(
        X_reduced[101][0],
        X_reduced[101][1],
        marker="s",
        markerfacecolor="green",
        markeredgecolor="black",
    )

    plt.autoscale()
    plt.savefig("lle.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    from min_dist import read_images_to_vector

    lle_vis()
