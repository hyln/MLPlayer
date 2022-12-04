import torch


def get_module_grad(module):
    grad = [param.grad.data.reshape(-1) for param in module.parameters()]
    grad = torch.cat(grad, -1)
    grad = grad.clone().detach()
    return grad


def set_module_grad(module, grad):
    for param in module.parameters():
        index = param.grad.data.numel()
        param.grad.data = grad[:index].reshape(param.grad.data.shape)
        grad = grad[index:]


def proj(u, v):
    # 将v向量标准化为单位向量
    v_std = torch.nn.functional.normalize(input=v, p=2, dim=-1)
    # 计算u,v向量的余弦相似度
    cos_sim = torch.cosine_similarity(x1=u, x2=v, dim=-1)
    # 计算u向量在v向量上的投影长度
    proj_len = torch.norm(input=u, p=2, dim=-1) * cos_sim
    # 计算u向量在v向量上的投影向量
    return v_std * proj_len


def angle(u, v):
    """计算两个向量的夹角"""
    angle = torch.arccos(torch.cosine_similarity(x1=u, x2=v, dim=-1))
    angle = angle * 180 / torch.pi
    return angle


if __name__ == "__main__":
    # from net import net
    #
    # grad = get_module_grad(net)
    # print(grad)

    g = torch.tensor([2.0, 2.0])
    v = torch.tensor([4.0, 0.0])
    print(g.shape)
    print(v.shape)
    vec = proj(g, v)
    print(vec)
