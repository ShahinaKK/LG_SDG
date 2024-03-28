from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from ccsdg.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from ccsdg.models.unet_ccsdg import UNetCCSDG, Projector
from ccsdg.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from ccsdg.datasets.utils.convert_csv_to_list import convert_labeled_list
from ccsdg.datasets.utils.transform import collate_fn_tr_styleaug, collate_fn_ts
from ccsdg.utils.lr import adjust_learning_rate
from ccsdg.utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
import torch.nn.functional as F
from einops import rearrange


def align_features(proj, tar_patches, slaug_textfeat):
    B, P, E = proj.shape
    PLoss_per_image = []
    NLoss_per_image = []

    for i in range(B):
        L_pos = []
        L_neg = []
        for j in range(P):
            pos_labels = []
            neg_labels = []
            patch_mask = tar_patches[i, j]
            patch_mask_labels = torch.unique(patch_mask)

            for k in range(1, 3):
                if k in patch_mask_labels:
                    pos_labels.append(k)
                else:
                    neg_labels.append(k)

            pos_labels_loss = []
            neg_labels_loss = []

            if len(pos_labels) == 0:
                L_pos.append(torch.tensor(0.0).cuda())
            else:
                for p in pos_labels:
                    plabel_text_feat = slaug_textfeat[p - 1].cuda()
                    cos_dist = torch.nn.functional.cosine_similarity(proj[i, j], plabel_text_feat, dim=0)
                    ploss_per_label = torch.max(torch.tensor(0.0), 1 - cos_dist)
                    pos_labels_loss.append(ploss_per_label)
                lpos_per_patch = torch.mean(torch.stack(pos_labels_loss, dim=0))
                L_pos.append(lpos_per_patch)

            if len(neg_labels) == 0:
                L_neg.append(torch.tensor(0.0).cuda())
            else:
                for n in neg_labels:
                    nlabel_text_feat = slaug_textfeat[n - 1].cuda()
                    cos_dist = torch.nn.functional.cosine_similarity(proj[i, j], nlabel_text_feat, dim=0)
                    nloss_per_label = torch.max(torch.tensor(0.0), 1 + cos_dist)
                    neg_labels_loss.append(nloss_per_label)

                lneg_per_patch = torch.mean(torch.stack(neg_labels_loss, dim=0))
                L_neg.append(lneg_per_patch)

        PLoss_per_image.append(sum(L_pos))
        NLoss_per_image.append(sum(L_neg))

    Loss_pos = torch.mean(torch.stack(PLoss_per_image, dim=0))
    Loss_neg = torch.mean(torch.stack(NLoss_per_image, dim=0))
    Loss_local = Loss_pos + Loss_neg

    return Loss_local



def train(args):
    model_name = args.model
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = args.save_interval
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle
    tau = 0.1

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, patch_size, img_normalize=False)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_tr_styleaug)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads//2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    model = UNetCCSDG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    projector = Projector()
    projector.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_prompt = torch.optim.SGD(list(projector.parameters()) + [model.channel_prompt],
                                       lr=initial_lr, momentum=0.99, nesterov=True)

    start_epoch = 0
    if continue_training:
        assert isfile(join(model_folder, 'model_latest.model')), 'missing model checkpoint!'
        params = torch.load(join(model_folder, 'model_latest.model'))
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    start = time()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))
        #####
        text_feat=torch.load("/l/users/20020135/CCSDG_difdot11.pt")
        #####
        train_loss_list = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        content_loss_list = list()
        style_loss_list = list()
        for iter, batch in enumerate(tr_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(batch['fda_data']).cuda().to(dtype=torch.float32) #[8,3,512,512]
            GLA_data = torch.from_numpy(batch['GLA']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)

            optimizer_prompt.zero_grad()
            with autocast():
                f_content, f_style = model.forward_first_layer(data, tau=tau)
                f_content_fda, f_style_fda = model.forward_first_layer(fda_data, tau=tau)#[8,64,256,256]
                f_content_GLA, f_style_GLA = model.forward_first_layer(GLA_data, tau=tau)
                f_content_p = projector(f_content)
                f_style_p = projector(f_style) #[8,1024]
                f_content_fda_p = projector(f_content_fda)
                f_style_fda_p = projector(f_style_fda)
                f_content_GLA_p = projector(f_content_GLA)
                f_style_GLA_p = projector(f_style_GLA)
                content_loss = F.l1_loss(f_content_p, f_content_fda_p, reduction='mean') + \
                               F.l1_loss(f_content_fda_p, f_content_p, reduction='mean') + \
                               F.l1_loss(f_content_p, f_content_GLA_p, reduction='mean') + \
                               F.l1_loss(f_content_GLA_p, f_content_p, reduction='mean') + \
                               F.l1_loss(f_content_GLA_p, f_content_fda_p, reduction='mean') + \
                               F.l1_loss(f_content_fda_p, f_content_GLA_p, reduction='mean')
                style_loss = F.l1_loss(f_style_p, f_style_fda_p, reduction='mean') + \
                             F.l1_loss(f_style_fda_p, f_style_p, reduction='mean') + \
                             F.l1_loss(f_style_p, f_style_GLA_p, reduction='mean') + \
                             F.l1_loss(f_style_GLA_p, f_style_p, reduction='mean') + \
                             F.l1_loss(f_style_GLA_p, f_style_fda_p, reduction='mean') + \
                             F.l1_loss(f_style_fda_p, f_style_GLA_p, reduction='mean')
                style_loss = - style_loss
            amp_grad_scaler.scale(content_loss + style_loss).backward()
            amp_grad_scaler.unscale_(optimizer_prompt)
            amp_grad_scaler.step(optimizer_prompt)
            amp_grad_scaler.update()

            optimizer.zero_grad()
            with autocast():
                proj,output = model(fda_data, tau=tau)
                loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output[:, 1], (seg[:, 0] == 2)*1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            optimizer.zero_grad()
            with autocast():
                proj, output = model(GLA_data, tau=tau)

                ###
                #proj=[8,512,16,16]
                # #seg=[8,1,512,512] 
                rproj = rearrange(proj,'b e h w -> b (h w) e ',h=16, w=16, e=512) #[8,256,512]
                rseg = rearrange(seg,'b c (h p) (w q) -> b c (p q) (h w)', p=16, q=16, h=32, w=32)#[8,1,256,1024]
                rseg = torch.mean(rseg, dim =1)
                loss_local = align_features(rproj, rseg, text_feat)
                loss_local =loss_local/256
                loss = loss_local + criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            optimizer.zero_grad()
            with autocast():
                proj, output = model(data, tau=tau)
                loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            train_loss_list.append(loss.detach().cpu().numpy())
            content_loss_list.append(content_loss.detach().cpu().numpy())
            style_loss_list.append(style_loss.detach().cpu().numpy())
            output_sigmoid = torch.sigmoid(output)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        mean_content_loss = np.mean(content_loss_list)
        mean_style_loss = np.mean(style_loss_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        writer.add_scalar("Train Scalars/Train Content Loss", mean_content_loss, epoch)
        writer.add_scalar("Train Scalars/Train Style Loss", mean_style_loss, epoch)
        writer.add_scalar("Train Scalars/Disc Dice", mean_tr_disc_dice, epoch)
        writer.add_scalar("Train Scalars/Cup Dice", mean_tr_cup_dice, epoch)
        print('  Tr loss: {}\n'
              '  Tr disc dice: {}; Cup dice: {}'.format(mean_tr_loss, mean_tr_disc_dice, mean_tr_cup_dice))

        val_loss_list = list()
        val_disc_dice_list = list()
        val_cup_dice_list = list()
        with torch.no_grad():
            model.eval()
            for iter, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                with autocast():
                    proj, output = model(data, tau=tau)
                    loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
                val_loss_list.append(loss.detach().cpu().numpy())
                output_sigmoid = torch.sigmoid(output)
                val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
        mean_val_loss = np.mean(val_loss_list)
        mean_val_disc_dice = np.mean(val_disc_dice_list)
        mean_val_cup_dice = np.mean(val_cup_dice_list)
        writer.add_scalar("Val Scalars/Val Loss", mean_val_loss, epoch)
        writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
        writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)
        writer.add_image('Val/Input', make_grid(data[:10], 10, normalize=True), epoch)
        writer.add_image('Val/Output Disc', make_grid(output_sigmoid[:10, 0][:, np.newaxis], 10, normalize=True), epoch)
        writer.add_image('Val/Output Cup', make_grid(output_sigmoid[:10, 1][:, np.newaxis], 10, normalize=True), epoch)
        writer.add_image('Val/Seg', make_grid(seg[:10], 10, normalize=True), epoch)

        print('  Val loss: {}\n'
              '  Val disc dice: {}; Cup dice: {}'.format(mean_val_loss, mean_val_disc_dice, mean_val_cup_dice))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
        if (epoch+1) % 200 == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format(epoch+1))
            torch.save(saved_model, join(model_folder, 'model_{}.model'.format(epoch+1)))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)
    saved_model = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    print('Saving model_{}.model...'.format('final'))
    torch.save(saved_model, join(model_folder, 'model_final.model'))
    if isfile(join(model_folder, 'model_latest.model')):
        os.remove(join(model_folder, 'model_latest.model'))
    total_time = time() - start
    print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))

    # inference
    from ccsdg.inference.inference_nets.inference_unet_ccsdg import inference
    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        inference('model_final.model', gpu, log_folder, patch_size, root_folder, [ts_csv_path], inference_tag, tau=tau)

