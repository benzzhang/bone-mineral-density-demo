from __future__ import print_function
from math import ceil

import os
import shutil
from symbol import file_input
import time
import yaml
import numpy as np
import nibabel

import torch
import torch.utils.data as data
import torch.optim as optim

import models
import dataset
from utils import Logger, AverageMeter, progress_bar
import losses
from losses import DiceCoeff
import cv2

state = {}
best_loss = np.Inf
use_cuda = False

def main(config_file):
    global state, best_loss, use_cuda

    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    if not os.path.exists(common_config['save_path']):
        os.makedirs(common_config['save_path'])

    title = 'Vertebrae Segmentation using' + common_config['arch']

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']]()
    model = torch.nn.DataParallel(model)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # optimizer and scheduler
    state['lr'] = common_config['lr']
    criterion = losses.__dict__[config['loss_config']['type']]()
    optimizer = optim.Adam(
       filter(
           lambda p: p.requires_grad,
           model.parameters()
           ),
        lr=common_config['lr'],
        weight_decay=common_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **common_config[common_config['scheduler_lr']])
    
    # initial dataset and dataloader 
    data_config = config['dataset']
    if args.infer:
        print('==> Preparing dataset %s' % data_config['type_infer'])
    else:
        print('==> Preparing dataset %s' % data_config['type'])

    # the path of inferring
    if args.infer:
        inferset = dataset.__dict__[data_config['type_infer']](
            os.path.join(os.path.dirname(os.path.realpath(__file__)), data_config['infer_list']),
            subset='inferred',
            prefix=data_config['infer_prefix']
            )

        inferloader = data.DataLoader(
            inferset, batch_size=common_config['infer_batch'], shuffle=False, num_workers=16)

        checkpoints = torch.load(os.path.join(common_config['save_path'], 'model_best.pth.tar'))
        model.load_state_dict(checkpoints['state_dict'], False)
        infer(inferloader, model, use_cuda, common_config)

        return

    # the path of training
    # logger
    logger = Logger(os.path.join(common_config['save_path'], 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Avg-Train Loss', 'Avg-Valid Loss'])

    # create dataset for training and testing
    trainset = dataset.__dict__[data_config['type']](
        os.path.join(os.path.dirname(os.path.realpath(__file__)), data_config['train_list']),
        subset='train',
        prefix=data_config['prefix']
        )
    validset = dataset.__dict__[data_config['type']](
        os.path.join(os.path.dirname(os.path.realpath(__file__)), data_config['valid_list']),
        subset='valid',
        prefix=data_config['prefix']
        )

    # create dataloader for training and testing
    trainloader = data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=16)
    validloader = data.DataLoader(
        validset, batch_size=common_config['valid_batch'], shuffle=False, num_workers=16)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True) if config['common']['fp16'] == True else None
    # Train and val
    for epoch in range(common_config['epoch']):
        print('\nEpoch: [%d | %d] LR: %f' %(epoch + 1, common_config['epoch'], state['lr']))
        train_loss, ep_train_dice = train(trainloader, model, criterion, optimizer, use_cuda, scaler, scheduler)
        valid_loss, ep_valid_dice = vaild(validloader, model, criterion, use_cuda, scaler)
        print('average train loss:  %.2f, train-%d dice: %.2f' %(train_loss, epoch + 1, ep_train_dice))
        print('average valid loss:  %.2f, valid-%d dice: %.2f' %(valid_loss, epoch + 1, ep_valid_dice))

        # save model
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)

        # append logger file
        logger.append([state['lr'], train_loss, valid_loss])
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path=common_config['save_path'])
        
    logger.close()
    print('best valid loss:' + str(best_loss))


def train(trainloader, model, criterion, optimizer, use_cuda, scaler=None, scheduler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    losses     = AverageMeter()
    end        = time.time()
    total_intersection = 0
    total_union = 0

    for batch_idx, datas in enumerate(trainloader):
        inputs, targets = datas

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        if scaler is None:
            outputs = model(inputs)
            loss, intersection, union = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        else:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss, intersection, union = criterion(outputs, targets)
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

        # if targets.sum() != 0:
        #     losses.update(loss.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f' % (loss))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_intersection += intersection
        total_union += union
    
    scheduler.step()
    epoch_dice = (2.0 * total_intersection + 1e-5) / (total_union + 1e-5)
    return losses.avg, epoch_dice


def vaild(validloader, model, criterion, use_cuda, scaler=None):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    total_intersection = 0
    total_union = 0

    for batch_idx, datas in enumerate(validloader):
        inputs, targets = datas

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss, intersection, union = criterion(outputs, targets)
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss, intersection, union = criterion(outputs, targets)

        # if targets.sum() != 0:
        #     losses.update(loss.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(validloader), 'Loss: %.2f' % (loss))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_intersection += intersection
        total_union += union
    
    epoch_dice = (2.0 * total_intersection + 1e-5) / (total_union + 1e-5)
    return losses.avg, epoch_dice

def infer(inferloader, model, use_cuda, common_config):
    # switch to infer mode
    model.eval()
    outputs = []

    for batch_idx, datas in enumerate(inferloader):
        # measure data loading time
        img, sagittal_len, transverse_len, img_name = datas

        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)
        with torch.no_grad():
            output = model(img)
            # output.shape - [64, 1, 512, 512]
            output = output.detach().cpu().numpy()
        outputs.append(output.astype(np.int8))
        progress_bar(batch_idx, len(inferloader))

    inferred_mask = np.concatenate([i for i in outputs], axis=0)    # inferred_mask.shape - [X, 1, 512, 512]
    inferred_mask = inferred_mask.transpose(1, 2, 3, 0)             # inferred_mask.shape - [1, 512, 512, X]
    inferred_mask = np.squeeze(inferred_mask)

    start = 0
    for s_len, t_len, name in zip (sagittal_len, transverse_len, img_name):
        s_len = s_len.unique().item()
        t_len = t_len.unique().item()
        name  = list(set(name))
        end = start + s_len
        instance_inferred_mask = inferred_mask[:, :, start:end].transpose(2, 0, 1)

        '''
        将分割结果中的错误部分(边缘的目标没能解决, 可能是边缘slice像素值分布区间过窄, 导致全部像素激活)去掉,只保留中心椎体部分:stupid method
        '''
        aligned_instance_inferred_mask = instance_inferred_mask[:, :, 0:t_len]
        cropped_instance_inferred_mask = np.zeros_like(aligned_instance_inferred_mask)
        Smin, Smax = int((end - start)*0.5-50), int((end - start)*0.5+50)
        Cmin, Cmax = int((end - start)*0.2), int((end - start)*0.8)
        cropped_instance_inferred_mask[Smin:Smax, Cmin:Cmax, :] = aligned_instance_inferred_mask[Smin:Smax, Cmin:Cmax, :]

        nft = nibabel.Nifti1Image(cropped_instance_inferred_mask, np.eye(4))
        save_folder = os.path.join(common_config['save_path'], 'inferred_results/')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        start = end
        
        file_name = str(name[0]).split('.')[0]+'_inferred.nii.gz'
        nibabel.save(nft, os.path.join(save_folder, file_name))
        print(file_name + ' saved ')
    print('infer succeed')
    return 


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, config):
    global state
    if epoch in config['scheduler']:
        state['lr'] *= config['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Vertebrae Segmentation')

    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str, default='experiments/template/config.yaml')
    parser.add_argument('--gpu-id', type=str, default='0,1,2,3')
    parser.add_argument('--infer', action='store_true')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)