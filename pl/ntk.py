import torch
from functorch import jacrev
from net import net_func, params, buffers, net
from log import writer
from memory_profiler import profile


def calc_jac_matrix(x):
    jac = jacrev(net_func)(params, buffers, x)
    jac = [j.flatten(1) for j in jac]
    return jac


def calc_ntk_matrix(jac_matrix):
    return torch.mm(jac_matrix, torch.transpose(jac_matrix, 0, 1))


def calc_global_ntk_matrix(jac_matrix):
    jac = torch.concat(jac_matrix, dim=1)
    return calc_ntk_matrix(jac)


def calc_local_ntk_matrix(jac_matrix):
    ntk_dict = {}
    i = -1
    for param in net.named_parameters():
        i += 1
        if param[0].find("conv") == -1:
            continue
        ntk_dict[param[0]] = calc_ntk_matrix(jac_matrix[i])
    return ntk_dict


def calc_miu_lf_cn(ntk_matrix):
    singular_values = torch.linalg.svdvals(ntk_matrix)
    singular_values_sorted, _ = torch.sort(singular_values)
    min_index = singular_values_sorted.nonzero()[0][0]
    return (
        singular_values_sorted[min_index],
        singular_values_sorted[-1],
        singular_values_sorted[-1] / singular_values_sorted[min_index],
    )


@profile
def log_ntk(x, global_step=0, add_scalars=True):
    jac = calc_jac_matrix(x)

    global_ntk = calc_global_ntk_matrix(jac)
    local_ntk = calc_local_ntk_matrix(jac)

    global_miu, global_lf, global_cn = calc_miu_lf_cn(global_ntk)

    local_miu_dict = {}
    local_lf_dict = {}
    local_cn_dict = {}
    for name, ntk in local_ntk.items():
        local_miu, local_lf, local_cn = calc_miu_lf_cn(ntk)
        local_miu_dict[name] = local_miu
        local_lf_dict[name] = local_lf
        local_cn_dict[name] = local_cn

    if not add_scalars:
        return

    writer.add_scalars(
        main_tag="global_miu_by_ntk",
        tag_scalar_dict={"global_miu": global_miu},
        global_step=global_step,
    )

    writer.add_scalars(
        main_tag="global_lf_by_ntk",
        tag_scalar_dict={"global_lf": global_lf},
        global_step=global_step,
    )

    writer.add_scalars(
        main_tag="global_cn_by_ntk",
        tag_scalar_dict={"global_cn": global_cn},
        global_step=global_step,
    )

    writer.add_scalars(
        main_tag="local_miu_by_ntk",
        tag_scalar_dict=local_miu_dict,
        global_step=global_step,
    )

    writer.add_scalars(
        main_tag="local_lf_by_ntk",
        tag_scalar_dict=local_lf_dict,
        global_step=global_step,
    )

    writer.add_scalars(
        main_tag="local_cn_by_ntk",
        tag_scalar_dict=local_cn_dict,
        global_step=global_step,
    )


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    x = torch.randn(32, 1, 28, 28)
    log_ntk(x, add_scalars=False)
    print("NTK Cost: {}s".format(time.perf_counter() - start))

    # jac = calc_jac_matrix(x)
    # from net import net

    # print(jac)
    # for j in jac:
    #     print(j.shape)

    # n = 0
    # for param in net.named_parameters():
    #     if param[0].find("conv") == -1:
    #         continue
    #     print(param[0])
    #     n += 1
    # print(n)

    # ntk_dict = calc_local_ntk_matrix(jac_matrix=jac)
    # print(ntk_dict)

    # print(jac)
    # for j in jac:
    #     print(j.shape)
    # torch.mm(jac, torch.transpose(jac, 0, 1))

    # jac = torch.concat(jac, dim=1)
    # print("jac2: ", jac)
    # for j in jac:
    #     print(j.shape)
    # # jac = [j.flatten(0) for j in jac]
    # # print("jac3: ", jac)

    # jac = calc_global_ntk_matrix(jac)
    # print(jac)
    #
    # miu, lf, cn = calc_miu_lf_cn(jac)
    # print(miu)
    # print(lf)
    # print(cn)
