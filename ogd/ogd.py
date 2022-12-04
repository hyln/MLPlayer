import torch
from util import proj


class OGD_Util:
    def __init__(self, norm_bound=1e-2):
        self.norm_bound = norm_bound  # 忽略norm小于norm_bound的梯度方向
        self.basis_vectors = []  # 保存basis vectors

    def save(
        self,
        dataloader,
        model,
        ogd_type="GTL",
        optimizer_func=torch.optim.SGD,
        learning_rate=1e-3,
        device=torch.device("cpu"),
    ):
        """
        保存模型在当前参数下的梯度向量, 用于分类任务
        :param model: nn.Module实例，返回值格式[batchsize, cls_num]的tensor
                      模型中应当有get_grad_vector()方法, 返回一个长为P的tensor向量, 其中P为模型的参数量
                      模型中应当有set_grad_vector(grad)方法, 将和get_grad_vector()得到的向量相同格式的梯度向量更新到模型中
        :param dataloader: torch.utils.data.DataLoader实例
        :param ogd_type: OGD类型可选择'ALL', ’AVE‘, 'GTL'
        :param device: 计算使用的设备, 默认cpu
        """
        model.eval()
        optimizer = optimizer_func(model.parameters(), lr=learning_rate)
        model = model.to(device)

        for _, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            batch_size = outputs.shape[0]
            cls_num = outputs.shape[1]
            if ogd_type == "ALL":
                for i in range(batch_size):
                    for j in range(cls_num):
                        outputs = model(inputs)
                        optimizer.zero_grad()
                        outputs[i][j].backward()
                        grad = model.get_grad_vector()
                        for v in self.basis_vectors:
                            grad = grad - proj(grad, v)
                        if torch.norm(grad, p=2, dim=-1) >= self.norm_bound:
                            self.basis_vectors.append(grad)
            elif ogd_type == "AVE":
                for j in range(cls_num):
                    grad = None
                    for i in range(batch_size):
                        outputs = model(inputs)
                        optimizer.zero_grad()
                        outputs[i][j].backward()
                        if i == 0:
                            grad = model.get_grad_vector()
                        else:
                            grad += model.get_grad_vector()
                    grad /= batch_size
                    for v in self.basis_vectors:
                        grad = grad - proj(grad, v)
                    if torch.norm(grad, p=2, dim=-1) >= self.norm_bound:
                        self.basis_vectors.append(grad)
            elif ogd_type == "GTL":
                for i in range(batch_size):
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    outputs[i][labels[i]].backward()
                    grad = model.get_grad_vector()
                    for v in self.basis_vectors:
                        grad = grad - proj(grad, v)
                    if torch.norm(grad, p=2, dim=-1) >= self.norm_bound:
                        self.basis_vectors.append(grad)

    def save_for_regression(
        self,
        dataloader,
        model,
        ogd_type="GTL",
        optimizer_func=torch.optim.SGD,
        learning_rate=1e-3,
        device=torch.device("cpu"),
    ):
        """
        保存模型在当前参数下的梯度向量, 用于回归任务
        :param model: nn.Module实例，返回值格式[batchsize, 1]的tensor
                      模型中应当有get_grad_vector()方法, 返回一个长为P的tensor向量, 其中P为模型的参数量
                      模型中应当有set_grad_vector(grad)方法, 将和get_grad_vector()得到的向量相同格式的梯度向量更新到模型中
        :param dataloader: torch.utils.data.DataLoader实例
        :param ogd_type: OGD类型可选择'ALL', ’AVE‘, 'GTL'
        :param device: 计算使用的设备, 默认cpu
        """
        model.eval()
        optimizer = optimizer_func(model.parameters(), lr=learning_rate)
        model = model.to(device)
        for _, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            batch_size = outputs.shape[0]
            if ogd_type == "ALL" or ogd_type == "GTL":
                for i in range(batch_size):
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    outputs[i].backward()
                    grad = model.get_grad_vector()
                    for v in self.basis_vectors:
                        grad = grad - proj(grad, v)
                    if torch.norm(grad, p=2, dim=-1) >= self.norm_bound:
                        self.basis_vectors.append(grad)
            elif ogd_type == "AVE":
                grad = None
                for i in range(batch_size):
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    outputs[i].backward()
                    if i == 0:
                        grad = model.get_grad_vector()
                    else:
                        grad += model.get_grad_vector()
                grad /= batch_size
                for v in self.basis_vectors:
                    grad = grad - proj(grad, v)
                if torch.norm(grad, p=2, dim=-1) >= self.norm_bound:
                    self.basis_vectors.append(grad)

    def orthogonalize(self, grad):
        # 将原始的方向修改为和basis vectors正交的方向
        for u in self.basis_vectors:
            grad -= proj(grad, u)
        return grad


if __name__ == "__main__":
    from dataset import task1_dataloader
    from net import net
    from linear import linear_model, task_dataloader

    # ogd_util1 = OGD_Util()
    # ogd_util1.save(dataloader=task1_dataloader, model=net, ogd_type="ALL")
    # ogd_util2 = OGD_Util()
    # ogd_util2.save(dataloader=task1_dataloader, model=net, ogd_type="AVE")
    # ogd_util3 = OGD_Util()
    # ogd_util3.save(dataloader=task1_dataloader, model=net, ogd_type="GTL")
    # print("ALL", len(ogd_util1.basis_vectors))
    # print("AVE", len(ogd_util2.basis_vectors))
    # print("GTL", len(ogd_util3.basis_vectors))

    ogd_util1 = OGD_Util()
    ogd_util1.save_for_regression(
        dataloader=task_dataloader, model=linear_model, ogd_type="ALL"
    )
    ogd_util2 = OGD_Util()
    ogd_util2.save_for_regression(
        dataloader=task_dataloader, model=linear_model, ogd_type="AVE"
    )
    ogd_util3 = OGD_Util()
    ogd_util3.save_for_regression(
        dataloader=task_dataloader, model=linear_model, ogd_type="GTL"
    )
    print("ALL", len(ogd_util1.basis_vectors))
    print("AVE", len(ogd_util2.basis_vectors))
    print("GTL", len(ogd_util3.basis_vectors))
    print("GTL", ogd_util3.basis_vectors)
