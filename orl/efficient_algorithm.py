import numpy as np
from pca_lda import calc_sw, calc_sb
from sympy.matrices import Matrix, GramSchmidt
from pca import pca, dim_reduction


# 计算st的零子空间的正交补空间, 即st的非零特征值对应的标准正交的特征向量
def calc_b_matrix(st):
    eigenvalues, eigenvectors = np.linalg.eigh(st)
    nonzero_indices = np.nonzero(np.around(eigenvalues))
    eigenvectors = eigenvectors[nonzero_indices]
    return eigenvectors.T


# 计算sb和st的prime矩阵
def calc_prime_matrix(sb, st, b):
    sb_prime = np.linalg.multi_dot([b.T, sb, b])
    st_prime = np.linalg.multi_dot([b.T, st, b])
    return sb_prime, st_prime


# 计算广义特征值和特征向量: 求解st_prime的逆和sb_prime相乘后的矩阵的最大特征值对应的单位特征向量η
def lda_efficient(st_prime, sb_prime):
    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(st_prime).dot(sb_prime))
    sort = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


# 计算原问题的解(α为鉴别向量): α = bη
def calc_alpha_matrix(b, eigenvectors, orthogonal=False):
    if not orthogonal:
        return np.dot(b, eigenvectors)

    # 正交化
    alpha = np.dot(b, eigenvectors)
    alpha_matrix = [Matrix(col) for col in alpha.T]
    gram = GramSchmidt(alpha_matrix)
    orthogonal_alpha_matrix = np.array(
        [np.array(g, dtype=np.float64).reshape(-1) for g in gram]
    ).T
    return orthogonal_alpha_matrix


# 投影到鉴别向量α
def lda_efficient_proj(inputs, alpha):
    return np.dot(inputs, alpha)


def validate(pca_dim=180, lda_dim=161):
    images = read_images_to_vector()

    # 先用PCA降维
    _, _, eigenvectors = pca(images)
    images = dim_reduction(images, eigenvectors, dim=pca_dim)

    # 拆分训练集和测试集, 并计算训练集的平均脸
    train_images, test_images = split_train_test(images)
    avg_faces = calc_avg_face(train_images=train_images)

    # 计算类内散度矩阵
    sw = calc_sw(train_images, avg_faces)
    # 计算类间散度矩阵
    sb = calc_sb(train_images, avg_faces)
    # 计算总散布矩阵
    st = sw + sb

    # 计算b矩阵
    b = calc_b_matrix(st)
    # 计算sb和st的prime矩阵
    sb_prime, st_prime = calc_prime_matrix(sb, st, b)
    # 计算广义特征值和特征向量
    eigenvalues, eigenvectors = lda_efficient(st_prime, sb_prime)

    # 计算鉴别向量α
    alpha = calc_alpha_matrix(b, eigenvectors[:, :lda_dim])

    # 将平均脸和测试样本都用LDA进行投影
    avg_faces_lda = lda_efficient_proj(avg_faces, alpha)
    test_images_lda = lda_efficient_proj(test_images, alpha)

    # 在测试集上测试
    validate_images(avg_faces_lda, test_images_lda)


if __name__ == "__main__":
    from min_dist import (
        read_images_to_vector,
        split_train_test,
        calc_avg_face,
        validate_images,
    )

    validate()
