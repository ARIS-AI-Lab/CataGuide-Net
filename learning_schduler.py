import math
import torch
from config import params
from torch.optim.lr_scheduler import _LRScheduler


class CosineLRSchedulerWithWarmupAndHold(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, hold_epochs,
                 initial_lr, warmup_initial_lr=1e-5,
                 min_lr=0.0, last_epoch=-1):
        """
        Cosine learning rate decay with Warmup and Hold period (Global Learning Rate).

        :param optimizer: Optimizer
        :param total_epochs: Total number of training epochs
        :param warmup_epochs: Number of Warmup epochs
        :param hold_epochs: Number of epochs to hold the learning rate constant after warmup
        :param initial_lr: Initial learning rate after warmup
        :param warmup_initial_lr: Initial learning rate for warmup (default: 1e-5)
        :param min_lr: Minimum learning rate (default: 0.0)
        :param last_epoch: The index of the last epoch. Default: -1.
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.hold_epochs = hold_epochs
        self.initial_lr = initial_lr
        self.warmup_initial_lr = warmup_initial_lr
        self.min_lr = min_lr

        # Validate epochs
        assert warmup_epochs + hold_epochs <= total_epochs, \
            "Warmup epochs and hold epochs should be less than or equal to total_epochs."

        super(CosineLRSchedulerWithWarmupAndHold, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1  # Current epoch (0-based indexing)

        if epoch <= self.warmup_epochs:
            # Warmup phase
            lr = self.warmup_initial_lr + (self.initial_lr - self.warmup_initial_lr) * (epoch / self.warmup_epochs)
        elif epoch <= self.warmup_epochs + self.hold_epochs:
            # Hold phase
            lr = self.initial_lr
        elif epoch <= self.total_epochs:
            # Cosine decay phase
            adjusted_epoch = epoch - self.warmup_epochs - self.hold_epochs
            adjusted_total_epochs = self.total_epochs - self.warmup_epochs - self.hold_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total_epochs))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        else:
            # After total_epochs, keep lr at min_lr
            lr = self.min_lr

        # Apply the same lr to all parameter groups
        return [lr for _ in self.optimizer.param_groups]
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

    optimizer = torch.optim.Adam(model.parameters(), lr=min_lr, weight_decay=params['model_param']['weight_decay'])
    scheduler = CosineLRSchedulerWithWarmupAndHold(optimizer, num_epochs, warmup_epochs, hold_epochs,
                                                   initial_lr=initial_lr, warmup_initial_lr=min_lr, min_lr=0)
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