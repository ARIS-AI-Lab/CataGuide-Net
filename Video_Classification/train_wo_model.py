import torch
import numpy as np
from src.utils.DataLoader import load_dataloader
from src.utils import metrics
from config import params
import os
from src.utils.loss import FocalLoss
from eval.evaluation import load_dataloader_eval
from model.Model import Model
from model.test_model import ResNet50TemporalModel
from src.learning_schduler import CosineLRSchedulerWithWarmupAndHold
from torch.utils.tensorboard import SummaryWriter
from core import utils,config_utils
import time
from tqdm import tqdm
from model.TimeSformer import TimeSformer


def load_data():
    train_dataloader = load_dataloader(mode='train')
    test_dataloader = load_dataloader_eval(mode='test')

    return train_dataloader, test_dataloader
    # return train_dataloader

def main():
    num_epochs = params['num_epoch']
    warmup_epochs = params['model_param']['warmup_epochs']
    hold_epochs = params['model_param']['hold_epochs']
    initial_lr = params['model_param']['max_learning_rate']
    min_lr = params['model_param']['base_learning_rate']
    save_interval = params['model_param']['save_checkpoint_epochs']
    model_save_path_ab = params['model_param']['model_dir']
    pretrain_path = params['model_param']['pretrain_model_path']
    n_batches_tr = n_batches_te = params['num_samples']
    saved_checkpoints = []

    train_dataloader, test_dataloader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(os.path.exists(params['model_param']['model_dir']))
    if not os.path.exists(params['model_param']['model_dir']):
        print(params['model_param']['model_dir'])
        os.makedirs(params['model_param']['model_dir'])

    torch.cuda.empty_cache()
    model = TimeSformer(
        dim=128,
        image_size=224,
        patch_size=28,
        num_frames=64,
        num_classes=10,
        depth=6,
        heads=4,
        dim_head=8,
        attn_dropout=0.2,
        ff_dropout=0.6,
        # rotary_emb=False,
    ).to(device)
    # model = ResNet50TemporalModel(num_classes=10, pretrained=True).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = FocalLoss()
    metric_fn = metrics.map_charades
    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr, weight_decay=params['model_param']['weight_decay'],
                                momentum=0.8)
    scheduler = CosineLRSchedulerWithWarmupAndHold(optimizer, num_epochs, warmup_epochs, hold_epochs,
                                                   initial_lr=initial_lr, warmup_initial_lr=min_lr, min_lr=1e-6)

    writer = SummaryWriter(log_dir=params['model_param']['model_dir'])

    # '''
    if not params['is_train_from_begin']:
        pretrained_dict = torch.load(pretrain_path)
        model_dict = model.state_dict()

        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and "layer" in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('model loaded from {}'.format(os.path.basename(pretrain_path)))

    default_config_file = 'charades_i3d_tc2_f256.yaml'
    config_file = default_config_file
    config_path = './configs/%s' % (config_file)
    config_utils.cfg_from_file(config_path)
    n_epochs = params['num_epoch']
    step_train = 0
    step_test = 0
    for idx_epoch in range(n_epochs):
        duration = 0.0
        epoch_num = idx_epoch + 1
        batch_num = 0
        loss_tr = 0.0
        acc_tr = 0.0
        loss_te = 0.0
        acc_te = 0.0


        tt1 = time.time()
        # scaler = torch.cuda.amp.GradScaler()
        # flag model as training
        model.train()

        # training
        for idx_batch, (x, y_true) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            batch_num = idx_batch + 1

            x, y_true = x.to(device), y_true.to(device)
            # mask = torch.ones(params['batch_size'], params['num_samples']).bool().to(device)
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            y_pred = model(x)
            y_true = y_true.long()
            print('y_pred:', y_pred.shape)
            print('x:', x.shape)
            # print('y_true:', y_true)
            # print("y_pred shape:", y_pred.shape)
            # print("y_true shape:", y_true.shape)
            # print("y_true dtype:", y_true.dtype)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            # print(f"y_true min: {y_true.min()}, max: {y_true.max()}")
            y_true = y_true.cpu().numpy().astype(np.int32)
            y_pred = y_pred.cpu().detach().numpy()
            print(y_true.shape)
            print(y_pred.shape)
            exit(0)
            # print(loss)
            loss_b_tr = loss.item()
            acc_b_tr = metric_fn(y_true, y_pred)

            loss_tr += loss_b_tr
            acc_tr += acc_b_tr
            # print(acc_tr)
            loss_b_tr = loss_tr / float(batch_num)
            acc_b_tr = acc_tr / float(batch_num)

            writer.add_scalar('Train/loss_tr', loss_b_tr, step_train)
            writer.add_scalar('Train/acc_tr', acc_b_tr, step_train)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, step_train)
            step_train += 1

            tt2 = time.time()
            duration = tt2 - tt1
        if (epoch_num) % save_interval == 0:
            # Define the path and filename for saving the model
            model_save_path = os.path.join(model_save_path_ab, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at epoch {epoch_num}')

            saved_checkpoints.append(model_save_path)

            if len(saved_checkpoints) > params['max_checkpoints']:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Deleted old checkpoint: {oldest_checkpoint}")

        scheduler.step()
        loss_b_tr = loss_tr / float(batch_num)
        acc_b_tr = acc_tr / float(batch_num)
        print('\r%04ds - epoch: %02d/%02d, batch [tr]: %02d/%02d, loss:%s, %s: %0.2f' % (
            duration, epoch_num, n_epochs, batch_num, n_batches_tr, loss_b_tr, 'map', acc_b_tr))


        # '''
        # flag model as testing
        if step_train % 20 == 0:
            model.eval()


            # testing
            for idx_batch, (x, y_true) in enumerate(test_dataloader):
                batch_num = idx_batch + 1

                x, y_true = x.to(device), y_true.to(device)
                y_pred = model(x)
                loss_b_te = loss_fn(y_pred, y_true).cpu().detach().numpy()
                y_true = y_true.cpu().numpy().astype(np.int32)
                y_pred = y_pred.cpu().detach().numpy()
                acc_b_te = metric_fn(y_true, y_pred)

                loss_te += loss_b_te
                acc_te += acc_b_te
                loss_b_te = loss_te / float(batch_num)
                acc_b_te = acc_te / float(batch_num)

                writer.add_scalar('Test/loss_tr', loss_tr, step_test)
                writer.add_scalar('Test/acc_tr', acc_tr, step_test)
                step_test += 1

                tt2 = time.time()
                duration = tt2 - tt1
                print('\r%04ds - epoch: %02d/%02d, batch [te]: %02d/%02d, loss, %s: %0.2f, %0.2f ' % (
                duration, epoch_num, n_epochs, batch_num, n_batches_te, 'map', loss_b_te, acc_b_te))

            loss_tr /= float(n_batches_tr)
            loss_te /= float(n_batches_te)
            acc_tr /= float(n_batches_tr)
            acc_te /= float(n_batches_te)

            tt2 = time.time()
            duration = tt2 - tt1
            print('\r%04ds - epoch: %02d/%02d, [tr]: %0.2f, %0.2f, [te]: %0.2f, %0.2f           \n' % (
            duration, epoch_num, n_epochs, loss_tr, acc_te, loss_te, acc_te))
    # '''
    writer.close()
    # '''


if __name__ == '__main__':
    main()