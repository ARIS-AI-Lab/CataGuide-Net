import math
import torch
from config import params
from torch.optim.lr_scheduler import _LRScheduler


class CosineLRSchedulerWithWarmupAndHold(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, hold_epochs, initial_lrs, warmup_initial_lrs=None,
                 min_lrs=None, last_epoch=-1):
        """
        Cosine learning rate decay with Warmup and Hold period.
        :param optimizer: Optimizer
        :param total_epochs: Total number of training epochs
        :param warmup_epochs: Number of Warmup epochs
        :param hold_epochs: Number of epochs to hold the learning rate constant after warmup
        :param initial_lrs: List of initial learning rates for each parameter group after warmup
        :param warmup_initial_lrs: List of initial learning rates for warmup (default is very small for all groups)
        :param min_lrs: List of minimum learning rates for each parameter group (default is 0 for all groups)
        :param last_epoch: The index of the last epoch. Default: -1.
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.hold_epochs = hold_epochs
        self.initial_lrs = initial_lrs
        self.warmup_initial_lrs = warmup_initial_lrs if warmup_initial_lrs is not None else [1e-5] * len(initial_lrs)
        self.min_lrs = min_lrs if min_lrs is not None else [0] * len(initial_lrs)
        super(CosineLRSchedulerWithWarmupAndHold, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(0, self.last_epoch + 1)
        lrs = []

        for base_lr, warmup_lr, min_lr in zip(self.initial_lrs, self.warmup_initial_lrs, self.min_lrs):
            if epoch == 0:
                lr = warmup_lr
            elif epoch < self.warmup_epochs:
                lr = warmup_lr + (base_lr - warmup_lr) * (epoch / self.warmup_epochs)
            elif epoch < self.warmup_epochs + self.hold_epochs:
                lr = base_lr
            else:
                adjusted_epoch = epoch - self.warmup_epochs - self.hold_epochs
                adjusted_total_epochs = self.total_epochs - self.warmup_epochs - self.hold_epochs
                if adjusted_total_epochs > 0:
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total_epochs))
                    lr = min_lr + (base_lr - min_lr) * cosine_decay
                else:
                    lr = min_lr
            lrs.append(lr)

        return lrs
# '''
# 示例用法
if __name__ == "__main__":
    import torch.optim as optim

    # 定义一个简单的模型和优化器
    model = torch.nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    num_epochs = params['num_epoch']
    warmup_epochs = params['model_param']['warmup_epochs']
    hold_epochs = params['model_param']['hold_epochs']
    initial_lr = params['model_param']['max_learning_rate']
    min_lr = params['model_param']['base_learning_rate']
    save_interval = params['model_param']['save_checkpoint_epochs']
    num_classes = len(params['model_param']['tool_and_eyes_class']) + 1  # class + background
    saved_checkpoints = []

    total_epochs = num_epochs

    optimizer = optim.SGD(model.parameters(), lr=min_lr, weight_decay=params['model_param']['weight_decay'])
    scheduler = CosineLRSchedulerWithWarmupAndHold(optimizer, num_epochs, warmup_epochs, hold_epochs, initial_lr,
                                                   warmup_initial_lr=min_lr, min_lr=0)
    # 模拟训练过程中逐轮更新学习率
    lr_values = []
    current_lr = optimizer.param_groups[0]['lr']
    print(current_lr)
    for epoch in range(total_epochs):
        # optimizer.zero_grad()
        # outputs = model(images)
        # outputs = model(images)
        #
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_values.append(current_lr)
        # epoch_loss = epoch_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, LR:{current_lr:.8f}')
        scheduler.step()
    import matplotlib.pyplot as plt
    plt.plot(range(1, total_epochs + 1), lr_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('CosineLR with Warmup and Hold')
    plt.grid(True)
    plt.show()
        # print(f"Epoch {epoch + 1}/{total_epochs}, Learning Rate: {lr:.6f}")
# '''