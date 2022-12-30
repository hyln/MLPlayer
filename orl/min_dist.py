import cv2
import glob
import numpy as np


# 读取图片
def read_images_to_vector(path=r".\data\ORL\*.bmp"):
    images = []
    for img in glob.glob(path):
        img = cv2.imread(img, 0)
        temp = np.resize(img, (img.shape[0] * img.shape[1]))
        images.append(temp.T)
    return np.array(images).astype("float64")


# 拆分训练集和测试集
def split_train_test(images):
    train_images = []
    test_images = []
    for i in range(40):
        [train_images.append(img) for img in images[i * 10 : i * 10 + 5]]
        [test_images.append(img) for img in images[i * 10 + 5 : i * 10 + 10]]
    return np.array(train_images), np.array(test_images)


# 计算平均脸
def calc_avg_face(train_images):
    avg_faces = []
    for i in range(40):
        avg_faces.append(np.mean(train_images[i * 5 : i * 5 + 5], axis=0))
    return np.array(avg_faces)


# 预测结果
def predict_image(avg_faces, image):
    min_index = 0
    min_value = np.linalg.norm(avg_faces[0] - image)
    for i in range(len(avg_faces)):
        value = np.linalg.norm(avg_faces[i] - image)
        if value <= min_value:
            min_index = i
            min_value = value
    return min_index


# 测试错误率
def validate_images(avg_faces, test_images):
    total = 0
    for i in range(40):
        err = 0
        for img in test_images[i * 5 : i * 5 + 5]:
            index = predict_image(avg_faces, img)
            err += index != i
        print("第{}类样本的错误数: {}".format(i + 1, err))
        total += err
    print("总共的错误数: {}".format(total))


if __name__ == "__main__":
    images = read_images_to_vector()
    train_images, test_images = split_train_test(images)
    avg_faces = calc_avg_face(train_images=train_images)
    validate_images(avg_faces, test_images)
