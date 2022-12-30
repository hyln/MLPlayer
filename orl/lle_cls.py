from sklearn.manifold import LocallyLinearEmbedding


if __name__ == "__main__":
    from min_dist import (
        read_images_to_vector,
        split_train_test,
        calc_avg_face,
        validate_images,
    )

    lle = LocallyLinearEmbedding(n_components=100, n_neighbors=5)

    images = read_images_to_vector()
    train_images, test_images = split_train_test(images)

    train_images = lle.fit_transform(train_images)
    test_images = lle.transform(test_images)

    avg_faces = calc_avg_face(train_images=train_images)
    validate_images(avg_faces, test_images)
