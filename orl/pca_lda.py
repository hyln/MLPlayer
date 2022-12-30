import numpy as np


# 计算类内散度矩阵
def calc_sw(train_images, avg_faces, sample=5):
    for i in range(train_images.shape[0]):
        train_images[i] -= avg_faces[int(i / sample)]
    sw = np.zeros((train_images.shape[1], train_images.shape[1]))
    for i in range(len(avg_faces)):
        sw += np.dot(
            train_images[i * sample : i * sample + sample].T,
            train_images[i * sample : i * sample + sample],
        )
    return sw / train_images.shape[0]


# 计算类间散度矩阵
def calc_sb(train_images, avg_faces, sample=5):
    avg_faces -= np.mean(train_images, axis=0)
    sb = np.zeros((train_images.shape[1], train_images.shape[1]))
    for i in range(len(avg_faces)):
        sb += np.dot(avg_faces[i].T, avg_faces[i])
    return sb * sample / train_images.shape[0]


# 计算广义特征值和特征向量
def lda(sw, sb):
    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(sw).dot(sb))
    sort = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


# 根据lda的结果进行投影
def lda_proj(inputs, eigenvectors, dim):
    return np.dot(inputs, eigenvectors[:, :dim])


def validate(pca_dim=50, lda_dim=39):
    images = read_images_to_vector()

    # 先用PCA降维
    _, _, eigenvectors = pca(images)
    images = dim_reduction(images, eigenvectors, dim=pca_dim)

    # 拆分训练集和测试集, 并计算训练集的平均脸
    train_images, test_images = split_train_test(images)
    avg_faces = calc_avg_face(train_images=train_images)

    # 计算LDA
    sw = calc_sw(train_images, avg_faces)
    sb = calc_sb(train_images, avg_faces)
    print(sw.shape)
    print(sb.shape)

    eigenvalues, eigenvectors = lda(sw, sb)

    # 将平均脸和测试样本都用LDA进行投影
    avg_faces = lda_proj(avg_faces, eigenvectors, dim=lda_dim)
    test_images = lda_proj(test_images, eigenvectors, dim=lda_dim)

    # 在测试集上测试
    validate_images(avg_faces, test_images)


if __name__ == "__main__":
    from min_dist import (
        read_images_to_vector,
        split_train_test,
        calc_avg_face,
        validate_images,
    )
    from pca import pca, dim_reduction

    validate()
