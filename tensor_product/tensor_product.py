import numpy as np


def calc_tensor_product(fillers, roles):
    res = np.zeros((len(fillers[0]), len(roles[0])))
    for i in range(len(fillers)):
        filler = np.expand_dims(fillers[i], axis=-1)
        role = np.expand_dims(roles[i], axis=0)
        res += np.dot(filler, role)
    return res


def query_name_by_id(h, id):
    return np.dot(h, id.T)


def query_id_by_name(name, h):
    return np.round(np.dot(name, h))


if __name__ == "__main__":
    # happy
    fillers = np.array(
        [[0, 1, 0, 3], [0, 1, 1, 1], [0, 0, 1, 2], [0, 0, 1, 2], [2, 2, 1, 1]]
    )
    roles = np.array(
        [
            [1, 1, 2, 1, 3],
            [2, 1, 1, 4, 5],
            [0, 0, 1, 2, 3],
            [1, 5, 2, 1, 0],
            [0, 4, 3, 2, 1],
        ]
    )
    res = calc_tensor_product(fillers, roles)
    print("happy: \n", res)
    sum = res

    # new
    fillers = np.array([[2, 0, 1, 0], [1, 2, 1, 1], [3, 2, 1, 0]])
    roles = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 4, 1, 2, 3],
            [0, 1, 0, 0, 2],
        ]
    )
    res = calc_tensor_product(fillers, roles)
    print("new: \n", res)
    sum += res

    # year
    fillers = np.array([[2, 2, 1, 1], [1, 2, 1, 1], [0, 1, 1, 1], [1, 2, 3, 4]])
    roles = np.array(
        [
            [1, 0, 0, 0, 0],
            [3, 2, 2, 1, 2],
            [2, 5, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    )
    res = calc_tensor_product(fillers, roles)
    print("year: \n", res)
    sum += res
    print("representation: \n", sum)

    # 第三题
    # 计算representation
    fillers = np.array(
        [
            [1, 5, 6, 1, 2, 4],
            [4, 3, 7, 3, 2, 1],
            [2, 1, 9, 0, 3, 7],
        ],
    )
    roles = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
    )
    h = calc_tensor_product(fillers, roles)
    print("presentation: \n", h)

    # query name by id
    name = query_name_by_id(h, roles[0])
    print("query name by id {}, name {}".format(roles[0], name))
    name = query_name_by_id(h, roles[1])
    print("query name by id {}, name {}".format(roles[1], name))
    name = query_name_by_id(h, roles[2])
    print("query name by id {}, name {}".format(roles[2], name))

    # query id by name
    # 计算u
    u = np.linalg.pinv(fillers.T)
    id = query_id_by_name(np.expand_dims(u[0], axis=0), h)
    print("query id by name {}, id {}".format(fillers[0], id))
    id = query_id_by_name(np.expand_dims(u[1], axis=0), h)
    print("query id by name {}, id {}".format(fillers[1], id))
    id = query_id_by_name(np.expand_dims(u[2], axis=0), h)
    print("query id by name {}, id {}".format(fillers[2], id))
