import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy as np
from utils import cal_performance, normalize_duration

from torch.utils.tensorboard import SummaryWriter


def train(args, model, train_loader, optimizer, scheduler, criterion, model_save_path, log_save_path, pad_idx, device):
    model.to(device)
    model.train()
    print("Training Start")
    
    writer = None
    if not args.ddp or device == 0:
        writer = SummaryWriter(log_save_path)

    for epoch in range(args.epochs):
        epoch_acc =0
        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_dur = 0
        epoch_loss_seg = 0
        total_class = 0
        total_class_correct = 0
        total_seg = 0
        total_seg_correct = 0

        print('epoch start %d' % epoch)
        for i, data in enumerate(train_loader):
            print('batch start %d' % i)
            optimizer.zero_grad()
            features, scene_graphs, past_label, trans_dur_future, trans_future_target = data
            features = features.to(device) #[B, S, C]
            scene_graphs = [sg.to(device) for sg in scene_graphs] #[T,B,F] a list of PyG Batches (1 per timestep)
            past_label = past_label.to(device) #[B, S]
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)

            B = trans_dur_future.size(0)
            target_dur = trans_dur_future*trans_dur_future_mask
            target = trans_future_target
            if args.input_type in ['i3d_transcript', 'nusc_bitmasks']:
                inputs = (features, None, past_label)
            elif args.input_type in ['nusc_bitmasks_scenegraphs', 'nusc_scenegraphs']:
                inputs = (features, scene_graphs, past_label)
            elif args.input_type == 'gt':
                gt_features = past_label.int()
                inputs = (gt_features, None, past_label)

            outputs = model(inputs)

            losses = 0
            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                #print('seg dims:', B, T, C)
                output_seg = output_seg.view(-1, C).to(device)
                target_past_label = past_label.view(-1)
                loss_seg, n_seg_correct, n_seg_total = cal_performance(output_seg, target_past_label, pad_idx)
                losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
            if args.anticipate :
                output = outputs['action']
                B, T, C = output.size()
                #print('antic dims:', B, T, C)
                output = output.view(-1, C).to(device)
                target = target.contiguous().view(-1)
                out = output.max(1)[1] #oneshot
                out = out.view(B, -1)
                loss, n_correct, n_total = cal_performance(output, target, pad_idx)
                acc = n_correct / n_total
                loss_class = loss.item()
                losses += loss
                total_class += n_total
                total_class_correct += n_correct
                epoch_loss_class += loss_class

                output_dur = outputs['duration']
                output_dur = normalize_duration(output_dur, trans_dur_future_mask)
                target_dur = target_dur * trans_dur_future_mask
                loss_dur = torch.sum(criterion(output_dur, target_dur)) / \
                torch.sum(trans_dur_future_mask)

                losses += loss_dur
                epoch_loss_dur += loss_dur.item()

            batch_loss = losses.item()
            epoch_loss += batch_loss
            losses.backward()
            optimizer.step()
            if writer:
                writer.add_scalar('Training Loss', batch_loss, epoch * len(train_loader) + i)

        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        if args.anticipate :
            accuracy = total_class_correct/total_class
            epoch_loss_class = epoch_loss_class / (i+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            epoch_loss_dur = epoch_loss_dur / (i+1)
            print('dur loss: %.5f'%epoch_loss_dur)

        if args.seg :
            acc_seg = total_seg_correct / total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)

        scheduler.step()

        if epoch >= 0 and (epoch + 1) % args.save_every == 0:
            save_file = os.path.join(model_save_path, 'checkpoint'+str(epoch)+'.ckpt')
            if args.ddp:
                state_dict = model.module.state_dict()
                if device == 0: #only save a checkpoint from one GPU
                    torch.save(state_dict, save_file)
            else:
                state_dict = model.state_dict()
                torch.save(state_dict, save_file)
        
        
    if writer:
        writer.close()


    return model

