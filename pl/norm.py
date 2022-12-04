import torch
from log import writer


def global_grad_norm_miu_by_def(params, loss):
    # grad_norm = sum([torch.pow(torch.norm(param.grad.data), 2) for param in params])
    grad_norm = sum([torch.norm(param.grad.data) for param in params])
    miu = grad_norm / loss
    return grad_norm, miu


def local_grad_norm_miu_by_def(param, loss):
    # grad_norm = torch.pow(torch.norm(param.grad.data), 2)
    grad_norm = torch.norm(param.grad.data)
    miu = grad_norm / loss
    return grad_norm, miu


def global_weight_norm_lf_by_def(params, loss):
    weight_norm = sum([torch.norm(param.data) for param in params])
    lf = loss / weight_norm
    return weight_norm, lf


def local_weight_norm_lf_by_def(param, loss):
    weight_norm = torch.norm(param.data)
    lf = loss / weight_norm
    return weight_norm, lf


def log_global_grad_norm_miu_by_def(params, loss, global_step):
    global_grad_norm, global_miu = global_grad_norm_miu_by_def(params, loss.item())
    writer.add_scalars(
        main_tag="global_grad_norm_by_def",
        tag_scalar_dict={
            "global_grad_norm": global_grad_norm.data.item(),
        },
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="global_miu_by_def",
        tag_scalar_dict={
            "global_miu": global_miu.data.item(),
        },
        global_step=global_step,
    )
    return global_miu


def log_local_grad_norm_miu_by_def(params, loss, global_step):
    norm_tag_scalar_dict = {}
    miu_tag_scalar_dict = {}
    for param in params:
        if len(param[1].shape) != 4:
            continue
        local_grad_norm, local_miu = local_grad_norm_miu_by_def(param[1], loss)
        norm_tag_scalar_dict[param[0]] = local_grad_norm.data.item()
        miu_tag_scalar_dict[param[0]] = local_miu.data.item()
    writer.add_scalars(
        main_tag="local_grad_norm_by_def",
        tag_scalar_dict=norm_tag_scalar_dict,
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="local_miu_by_def",
        tag_scalar_dict=miu_tag_scalar_dict,
        global_step=global_step,
    )


def log_global_weight_norm_lf_by_def(params, loss, global_step):
    global_weight_norm, global_lf = global_weight_norm_lf_by_def(params, loss.item())
    writer.add_scalars(
        main_tag="global_weight_norm_by_def",
        tag_scalar_dict={
            "global_weight_norm": global_weight_norm.data.item(),
        },
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="global_lf_by_def",
        tag_scalar_dict={
            "global_lf": global_lf.data.item(),
        },
        global_step=global_step,
    )
    return global_lf


def log_local_weight_norm_lf_by_def(params, loss, global_step):
    norm_tag_scalar_dict = {}
    lf_tag_scalar_dict = {}
    for param in params:
        if len(param[1].shape) != 4:
            continue
        local_weight_norm, local_lf = local_weight_norm_lf_by_def(param[1], loss)
        norm_tag_scalar_dict[param[0]] = local_weight_norm.data.item()
        lf_tag_scalar_dict[param[0]] = local_lf.data.item()
    writer.add_scalars(
        main_tag="local_weight_norm_by_def",
        tag_scalar_dict=norm_tag_scalar_dict,
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="local_lf_by_def",
        tag_scalar_dict=lf_tag_scalar_dict,
        global_step=global_step,
    )


def log_local_cn_by_def(params, loss, global_step):
    grad_norm_tag_scalar_dict = {}
    miu_tag_scalar_dict = {}
    weight_norm_tag_scalar_dict = {}
    lf_tag_scalar_dict = {}
    cn_tag_scalar_dict = {}
    for param in params:
        if param[0].find("conv") == -1:
            continue
        local_grad_norm, local_miu = local_grad_norm_miu_by_def(param[1], loss)
        grad_norm_tag_scalar_dict[param[0]] = local_grad_norm.data.item()
        miu_tag_scalar_dict[param[0]] = local_miu.data.item()
        local_weight_norm, local_lf = local_weight_norm_lf_by_def(param[1], loss)
        weight_norm_tag_scalar_dict[param[0]] = local_weight_norm.data.item()
        lf_tag_scalar_dict[param[0]] = local_lf.data.item()
        cn_tag_scalar_dict[param[0]] = local_lf / local_miu
    writer.add_scalars(
        main_tag="local_grad_norm_by_def",
        tag_scalar_dict=grad_norm_tag_scalar_dict,
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="local_miu_by_def",
        tag_scalar_dict=miu_tag_scalar_dict,
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="local_weight_norm_by_def",
        tag_scalar_dict=weight_norm_tag_scalar_dict,
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="local_lf_by_def",
        tag_scalar_dict=lf_tag_scalar_dict,
        global_step=global_step,
    )
    writer.add_scalars(
        main_tag="local_cn_by_def",
        tag_scalar_dict=cn_tag_scalar_dict,
        global_step=global_step,
    )


if __name__ == "__main__":
    from net import net

    for param in net.named_parameters():
        # weight_norm, lf = local_weight_norm_lf_by_def(param, 1)
        if param[0].find("conv") == -1:
            continue
        weight_norm, lf = local_weight_norm_lf_by_def(param[1], 1)
        print(weight_norm)
        print(lf)

    # weight_norm, lf = global_weight_norm_lf_by_def(net.parameters(), 1)
    # print(weight_norm)
    # print(lf)
    # import numpy as np
    # v = torch.from_numpy(np.array([[1., 2., 2.], [1., 2., 2.]]))
    # print(torch.norm(v))
    # grad_norm, miu = global_grad_norm_miu(v, 3.)
    # print(grad_norm)
    # print(miu)
