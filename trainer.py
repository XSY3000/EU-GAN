import os.path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel
import mydataset
from loss import *
from networks import network


class Trainer:
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.config = config
        self.proj_NAME = config.proj_NAME
        self.batch_size = config.batch_size
        self.iters = config.iters
        self.lr = config.lr
        self.logdir = config.logdir
        self.logger = config.tensorboard
        self.log = config.logger.print
        self.l1lossWeight = config.l1lossWeight
        self.BCELossWeight = config.BCELossWeight
        self.DiceLossWeight = config.DiceLossWeight
        self.PerceptualLossWeight = config.PerceptualLossWeight

        self.generatorG = getattr(network, config.generatorG['type'])()
        self.log(self.generatorG)
        self.generatorG.to(self.device)
        self.optimizer_GG = optim.AdamW(self.generatorG.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.lr_scheduler_GG = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_GG, T_max=config.iters,
                                                                    eta_min=1e-5)

        self.discriminatorG = getattr(network, config.discriminatorG['type'])()
        self.log(self.discriminatorG)
        self.discriminatorG.to(self.device)
        self.optimizer_DG = optim.AdamW(self.discriminatorG.parameters(), lr=self.lr * 0.1, betas=(0.9, 0.999))
        self.lr_scheduler_DG = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_DG, T_max=config.iters,
                                                                    eta_min=1e-5)
        self.current_iters = 0
        if config.resume is not None:
            for weight in config.resume:
                if 'generator' in weight:
                    self.generatorG.load(weight)
                    self.log('load model from {}'.format(weight))
                elif 'discriminator' in weight:
                    self.discriminatorG.load(weight)
                    self.log('load model from {}'.format(weight))
            self.current_iters = int(config.resume[0].split('_')[-1][:-4])

        # self.styleLoss = network.StyleLoss()
        # self.styleLoss.to(self.device)
        # self.styleLoss.eval()
        self.perceptualLoss = network.PerceptualLoss()
        self.perceptualLoss.to(self.device)
        self.perceptualLoss.eval()

    def train(self):
        self.train_dataset = getattr(mydataset, self.config.dataset)(mode='train', **self.config.traindataset_args)
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.BCELoss()
        dice_loss = DiceLoss()
        # l1_loss = torch.nn.L1Loss()
        best_loss = 1e10
        best_iter = 1
        log_lossGadv = []
        log_lossDadv = []
        log_lossPerceptual = []
        log_lossReconstruction = []
        log_lossdice = []
        log_lossminus = []
        log_lossbce = []
        log_recall = []
        times = []

        while self.current_iters <= self.iters:
            for images, labels, masks in dataloader:
                start_time = time.time()
                # if sum(rebuild_dataloader):
                #     dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
                self.current_iters += 1
                # images = images.to(self.device)
                # labels = labels.to(self.device)
                self.generatorG.train()

                # forward G
                G_pred = self.generatorG(images)

                # 训练判别器1
                self.discriminatorG.unfreeze()
                self.optimizer_DG.zero_grad()
                output_real = self.discriminatorG(labels)
                loss_real = criterion(output_real, torch.ones_like(output_real).to(self.device))
                output_fake = self.discriminatorG(G_pred.detach())
                loss_fake = criterion(output_fake, torch.zeros_like(output_fake).to(self.device))

                loss_Dadv = loss_real + loss_fake
                loss_Dadv.backward()
                log_lossDadv.append(loss_Dadv.item())
                self.optimizer_DG.step()
                self.discriminatorG.freeze()

                # 训练生成器
                self.optimizer_GG.zero_grad()

                output_fake = self.discriminatorG(G_pred)
                loss_Gadv = 0.01 * criterion(output_fake, torch.ones_like(output_fake).to(self.device))
                loss_Grec = self.BCELossWeight * criterion(G_pred * (1 - images), masks)
                # loss_Grec = self.BCELossWeight * criterion(torch.abs(labels-G_pred), torch.zeros_like(G_pred).to(self.device))
                loss_minus, recall = dice_loss(G_pred * (1 - images), masks)
                # loss_minus, recall = dice_loss(torch.abs(labels-G_pred), torch.zeros_like(G_pred).to(self.device))
                loss_minus *= self.DiceLossWeight
                loss_BCE = self.BCELossWeight * criterion(G_pred, labels)
                loss_dice = self.DiceLossWeight * dice_loss(G_pred, labels)[0]
                loss_Perceptual = self.PerceptualLossWeight * self.perceptualLoss(G_pred, labels)
                loss_G = loss_Grec + loss_Perceptual + loss_minus + loss_dice + loss_BCE + loss_Gadv
                loss_G.backward()
                log_lossGadv.append(loss_Gadv.item())
                log_lossReconstruction.append(loss_Grec.item())
                log_lossPerceptual.append(loss_Perceptual.item())
                log_lossminus.append(loss_minus.item())
                log_lossdice.append(loss_dice.item())
                log_lossbce.append(loss_BCE.item())
                log_recall.append(recall.item())

                self.optimizer_GG.step()

                if self.current_iters % 500 == 0:
                    # img = images[0] * torch.as_tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(
                    #     self.device) + torch.as_tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                    img = images[0].detach().cpu().numpy().squeeze()
                    # seg = images[0, 0].detach().cpu().numpy().squeeze()
                    label = labels[0].detach().cpu().numpy().squeeze()
                    pred_G = G_pred[0].detach().cpu().numpy().squeeze()
                    mask = masks[0].detach().cpu().numpy().squeeze()

                    # 设置画布像素大小
                    plt.rcParams['figure.figsize'] = (8.0, 4.0)
                    plt.rcParams['image.interpolation'] = 'bilinear'
                    # 更改dpi
                    plt.rcParams['figure.dpi'] = 600
                    plt.figure()
                    plt.subplot(1, 5, 1)
                    plt.imshow(img, cmap='gray')
                    plt.title('input')
                    plt.subplot(1, 5, 2)
                    plt.imshow(pred_G, cmap='gray')
                    plt.axis('off')
                    plt.title('pred')
                    plt.subplot(1, 5, 3)
                    plt.imshow(label, cmap='gray')
                    plt.axis('off')
                    plt.title('label')
                    plt.subplot(1, 5, 4)
                    plt.imshow(pred_G * (1 - img), cmap='gray')
                    plt.axis('off')
                    plt.title('minus')
                    plt.subplot(1, 5, 5)
                    plt.imshow(mask, cmap='gray')
                    plt.axis('off')
                    plt.title('mask')
                    # plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.1, left=0.1, right=0.9)
                    # 保存图片
                    figpath = os.path.join(self.logdir, f'figure/{self.current_iters}.png')
                    if not os.path.exists(os.path.dirname(figpath)):
                        os.makedirs(os.path.dirname(figpath))
                    plt.savefig(figpath)
                    # plt.show()
                    plt.close()

                times.append(time.time() - start_time)
                # 打印损失
                if self.current_iters % 20 == 0:
                    log_lossDadv = np.mean(log_lossDadv)
                    log_lossGadv = np.mean(log_lossGadv)
                    log_lossReconstruction = np.mean(log_lossReconstruction)
                    log_lossminus = np.mean(log_lossminus)
                    log_lossPerceptual = np.mean(log_lossPerceptual)
                    log_lossdice = np.mean(log_lossdice)
                    log_lossbce = np.mean(log_lossbce)
                    log_recall = np.mean(log_recall)
                    times = np.mean(times)
                    remaining_time = (self.iters - self.current_iters) * times
                    d, remaining_time = divmod(remaining_time, 24 * 3600)
                    h, remaining_time = divmod(remaining_time, 3600)
                    m, s = divmod(remaining_time, 60)

                    lr = self.optimizer_GG.param_groups[0]['lr']

                    self.log(
                        f' Iter: {self.current_iters}/{self.iters},'
                        f' Loss_Dadv: {log_lossDadv:.4f},'
                        f' Loss_Gadv: {log_lossGadv:.4f},'
                        f' Loss_bce: {log_lossReconstruction:.4f},'
                        f' Loss_dice: {log_lossminus:.4f},'
                        f' Loss_BCE: {log_lossbce:.4f},'
                        f' Loss_Dice: {log_lossdice:.4f},'
                        f' Loss_Perceptual: {log_lossPerceptual:.4f},'
                        f' Recall: {log_recall:.4f},'
                        f' lr: {lr:.6f},'
                        f' cost: {times:.4f}s,'
                        f' remain: ' + (f'{int(d):02d}day' if d else '') + f'{int(h):02d}:{int(m):02d}:{int(s):02d}'
                    )
                    self.logger.add_scalar('Loss_Dadv', log_lossDadv, self.current_iters)
                    self.logger.add_scalar('Loss_Gadv', log_lossGadv, self.current_iters)
                    self.logger.add_scalar('Loss_Grec', log_lossReconstruction, self.current_iters)
                    self.logger.add_scalar('Loss_Perceptual', log_lossPerceptual, self.current_iters)
                    self.logger.add_scalar('Loss_minus', log_lossminus, self.current_iters)
                    self.logger.add_scalar('Loss_dice', log_lossdice, self.current_iters)
                    self.logger.add_scalar('Loss_bce', log_lossbce, self.current_iters)
                    self.logger.add_scalar('Recall', log_recall, self.current_iters)
                    self.logger.add_scalar('lr', lr, self.current_iters)

                    log_lossDadv = []
                    log_lossGadv = []
                    log_lossReconstruction = []
                    log_lossPerceptual = []
                    log_lossminus = []
                    log_lossdice = []
                    log_lossbce = []
                    log_recall = []
                    times = []

                self.lr_scheduler_DG.step()
                self.lr_scheduler_GG.step()

                if self.current_iters >= self.iters:
                    break

                # eval
                if (self.current_iters % 500 == 0 and self.current_iters >= self.iters * 0.8) or self.current_iters == self.iters:
                    # if True:
                    loss = self.eval()
                    # loss = 0
                    self.log(f'Iter: {self.current_iters}, Loss: {loss}')
                    if loss < best_loss:
                        best_loss = loss
                        best_iter = self.current_iters
                    self.logger.add_scalar('eval_loss', loss, self.current_iters)
                    # save model
                    if not os.path.exists(os.path.join(self.logdir, f'checkpoints')):
                        os.makedirs(os.path.join(self.logdir, f'checkpoints'))
                    torch.save(self.generatorG.state_dict(),
                               os.path.join(self.logdir, f'checkpoints/generatorG_{self.current_iters}.pth'))
                    # torch.save(self.discriminatorG.state_dict(),
                    #            os.path.join(self.logdir, f'checkpoints/discriminatorG_{self.current_iters}.pth'))
                    self.log(f'Best iter: {best_iter}, Best loss: {best_loss}')

                # self.test()
        self.logger.close()

    def eval(self):
        self.val_dataset = getattr(mydataset, self.config.dataset)(mode='val', **self.config.valdataset_args)
        dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)
        self.generatorG.eval()
        criterion = DiceLoss()
        losses = 0
        recalls = 0
        with torch.no_grad():
            for images, labels, masks in tqdm.tqdm(dataloader):
                outputs = self.generatorG(images)
                loss, recall = criterion(outputs, labels)
                losses += loss
                recalls += recall
            losses = losses / len(dataloader)
            recalls = recalls / len(dataloader)
            self.log(f'Eval dice loss: {losses}, recall: {recalls}')
            return losses

    def test(self, save_dir=None, weight=None):
        if save_dir is not None:
            if save_dir == '':
                save_dir = os.path.join(self.logdir, f'output/test')
            elif '.' in save_dir:
                print(f'Error: {save_dir} is not a directory.')
                return
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if weight is not None:
            self.generatorG.load(weight, self.log)
        self.generatorG.eval()
        # TPs, FPs, TNs, FNs = [], [], [], []
        benchmark_Matrice = Matrice()
        global_Matrice = Matrice()
        minus_Matrice = Matrice()
        image_labeled_nums = 0
        output_labeled_nums = 0
        label_labeled_nums = 0
        self.test_dataset = getattr(mydataset, self.config.dataset)(mode='test', **self.config.testdataset_args)
        tq = tqdm.tqdm(self.test_dataset)
        with torch.no_grad():
            for input, label, mask, imagepath, image, (x_min, x_max, y_min, y_max) in tq:

                output = self.generatorG(input)
                output = torch.threshold(output, 0.75, 0)
                output = output * (1 - input) + input
                for i in range(4):
                    output = self.generatorG(output)
                    output = torch.threshold(output, 0.75, 0)
                output = output * (1 - input) + input
                output = F.interpolate(output, (y_max - y_min, x_max - x_min), mode='bilinear')
                output = F.pad(output, (x_min, 2024 - x_max, y_min, 3400 - y_max), 'constant', 0)
                output = torch.threshold(output, 0.5, 0) + torch.from_numpy(image).to(self.device).unsqueeze(
                    0).unsqueeze(0)
                output = output[:, :, 100:-100, 100:-100]
                output = self.generatorG(output)
                output = torch.threshold(output, 0.5, 0)
                output = self.generatorG(output)
                # output = torch.threshold(output, 0.5, 0)
                # output = self.generatorG(output)
                # output = torch.threshold(output, 0.5, 0)
                # output = self.generatorG(output)
                # output = torch.threshold(output, 0.5, 0)
                # output = self.generatorG(output)

                output = output.cpu().numpy().squeeze()
                # image = image.cpu().numpy().squeeze().astype(np.uint8)
                output = cv2.threshold(output, 0.5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
                output = np.pad(output, ((100, 100), (100, 100)), 'constant', constant_values=0)
                # output = cv2.erode(output, np.ones((3, 3)), iterations=1)
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('Goutput')
                # plt.show()
                output = output * (1 - cv2.dilate(image, np.ones((3, 3)), iterations=2))
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('326')
                # plt.show()
                output_mask = cv2.dilate(output, np.ones((7, 7)), iterations=3)
                # plt.imshow(output_mask[950:1150,400:600], 'gray')
                # plt.title('output_mask')
                # plt.show()
                output += image
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('334')
                # plt.show()
                output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((7, 7)), iterations=1)
                output = cv2.dilate(output, np.ones((3, 3)), iterations=1)
                # # plt.imshow(output[950:1150,400:600], 'gray')
                # # plt.title('339')
                # # plt.show()
                output = cv2.blur(output, (3, 3))
                # # plt.imshow(output[950:1150,400:600], 'gray')
                # # plt.title('343')
                # # plt.show()
                output = cv2.erode(output, np.ones((3, 3)), iterations=1)
                # # plt.imshow(output[950:1150,400:600], 'gray')
                # # plt.title('347')
                # # plt.show()
                output = output * output_mask
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('353')
                # plt.show()
                output = cv2.dilate(output, np.ones((3, 3)), iterations=3)
                output = cv2.blur(output, (9, 9))
                output = cv2.erode(output, np.ones((3, 3)), iterations=2)
                # output = output * (1 - image)
                # output = cv2.erode(output, np.ones((3, 3)), iterations=1)
                # output = output * (1 - image // 255)
                # output = cv2.dilate(output, np.ones((3, 3)), iterations=1) * 255
                output[output > 1] = 1
                output[image > 0] = 1
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('output')
                # plt.show()
                num, labels, stats, centroids = cv2.connectedComponentsWithStats(output, connectivity=8)
                for i in range(1, num):
                    if stats[i][-1] < 20:
                        output[labels == i] = 0
                # output = output * (1 - image) + image
                image *= 255
                output *= 255
                output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)[1]
                # output = cv2.resize(output, (3648, 5472))
                # image = cv2.resize(image, (3648, 5472))
                # label = cv2.resize(label, (3648, 5472))
                # mask = cv2.resize(mask, (3648, 5472))
                global_Matrice.update(output, label)
                benchmark_Matrice.update(image, label)
                minus_Matrice.update(output - image, mask)
                labeled_image, img_labeled_num = convert_labels_to_rgb(image)
                image_labeled_nums += img_labeled_num
                labeled_output, output_labeled_num = convert_labels_to_rgb(output)
                output_labeled_nums += output_labeled_num
                labeled_label, label_labeled_num = convert_labels_to_rgb(label)
                label_labeled_nums += label_labeled_num
                if save_dir is not None:
                    if '0610' in imagepath or '0618' in imagepath:
                        date = '0610_' if '0610' in imagepath else '0618_'
                    else:
                        date = ''
                    savepath = os.path.join(save_dir, 'rgb', date + os.path.basename(imagepath))
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, cv2.merge([image, label, output]))
                    savepath = savepath.replace('rgb', 'rw')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath,
                                cv2.merge([cv2.bitwise_and(image, output), cv2.bitwise_and(image, output), output]))
                    savepath = savepath.replace('rw', 'pred')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, output)
                    savepath = savepath.replace('pred', 'input')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, image)
                    savepath = savepath.replace('input', 'labeled_image')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, labeled_image)
                    savepath = savepath.replace('labeled_image', 'labeled_output')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, labeled_output)
                    savepath = savepath.replace('labeled_output', 'labeled_label')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, labeled_label)

                    # # 设置画布像素大小
                    # plt.rcParams['figure.figsize'] = (8.0, 4.0)
                    # plt.rcParams['image.interpolation'] = 'bilinear'
                    # # 更改dpi
                    # plt.rcParams['figure.dpi'] = 500
                    # plt.figure()
                    # plt.subplot(1, 5, 1)
                    # plt.imshow(image, cmap='gray')
                    # plt.title('input')
                    # plt.subplot(1, 5, 2)
                    # plt.imshow(output, cmap='gray')
                    # plt.axis('off')
                    # plt.title('pred')
                    # plt.subplot(1, 5, 3)
                    # plt.imshow(label, cmap='gray')
                    # plt.axis('off')
                    # plt.title('label')
                    # plt.subplot(1, 5, 4)
                    # plt.imshow(output - image, cmap='gray')
                    # plt.axis('off')
                    # plt.title('minus')
                    # plt.subplot(1, 5, 5)
                    # plt.imshow(mask, cmap='gray')
                    # plt.axis('off')
                    # plt.title('mask')
                    # # plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.1, left=0.1, right=0.9)
                    # # 保存图片
                    # figpath = os.path.join(self.logdir, f'figure/test/{tq.n}.png')
                    # if not os.path.exists(os.path.dirname(figpath)):
                    #     os.makedirs(os.path.dirname(figpath))
                    # plt.savefig(figpath)
                    # # plt.show()
                    # plt.close()
                    # TP, FP, TN, FN = calculate_confusion_matrix(pred, label_org)
                    # TPs.append(TP)
                    # TNs.append(TN)
                    # FPs.append(FP)
                    # FNs.append(FN)
                    # TP = np.sum(TPs)
                    # FP = np.sum(FPs)
                    # TN = np.sum(TNs)
                    # FN = np.sum(FNs)
                    # accuracy = (TP + TN) / (TP + TN + FP + FN)
                    # precision = TP / (TP + FP)
                    # recall = TP / (TP + FN)
                    # iou = TP / (TP + FP + FN)
                    # dice = 2 * TP / (2 * TP + FP + FN)
            accuracy, precision, recall, iou, dice = benchmark_Matrice.compute()
            self.log(
                f'benchmark:  acc: {accuracy}, precision: {precision}, recall: {recall}, iou: {iou}, dice: {dice}')
            accuracy, precision, recall, iou, dice = global_Matrice.compute()
            self.log(
                f'global:  acc: {accuracy}, precision: {precision}, recall: {recall}, iou: {iou}, dice: {dice}')
            self.log(
                f'global: TP: {global_Matrice.TP}, FP: {global_Matrice.FP}, TN: {global_Matrice.TN}, FN: {global_Matrice.FN}')
            accuracy, precision, recall, iou, dice = minus_Matrice.compute()
            self.log(
                f'minus:  acc: {accuracy}, precision: {precision}, recall: {recall}, iou: {iou}, dice: {dice}')
            self.log(
                f'minus: TP: {minus_Matrice.TP}, FP: {minus_Matrice.FP}, TN: {minus_Matrice.TN}, FN: {minus_Matrice.FN}'
            )
            self.log(
                f'image_labeled_nums: {image_labeled_nums}, output_labeled_nums: {output_labeled_nums}, label_labeled_nums: {label_labeled_nums}')

    def infer(self, image_path, save_dir=None, weights=None):
        if save_dir is None:
            save_dir = os.path.join(self.logdir, f'infer')
        if weights is not None:
            self.generatorG.load(weights)
        self.generatorG.eval()
        infer_dataset = mydataset.CustomDataset_G(image_path, mode='infer')
        tq = tqdm.tqdm(infer_dataset)
        with torch.no_grad():
            for input, imagepath, image, (x_min, x_max, y_min, y_max) in tq:

                output = self.generatorG(input)
                output = torch.threshold(output, 0.75, 0)
                output = output * (1 - input) + input
                for i in range(1):
                    output = self.generatorG(output)
                    output = torch.threshold(output, 0.75, 0)
                output = output * (1 - input) + input
                output = F.interpolate(output, (y_max - y_min, x_max - x_min), mode='bilinear')
                output = F.pad(output, (x_min, 2024 - x_max, y_min, 3400 - y_max), 'constant', 0)
                output = torch.threshold(output, 0.5, 0) + torch.from_numpy(image).to(self.device).unsqueeze(
                    0).unsqueeze(0)
                output = output[:, :, 100:-100, 100:-100]
                output = self.generatorG(output)
                # output = torch.threshold(output, 0.5, 0)
                # output = self.generatorG(output)
                # output = torch.threshold(output, 0.5, 0)
                # output = self.generatorG(output)

                output = output.cpu().numpy().squeeze()
                # image = image.cpu().numpy().squeeze().astype(np.uint8)
                output = cv2.threshold(output, 0.5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
                output = np.pad(output, ((100, 100), (100, 100)), 'constant', constant_values=0)
                # output = cv2.erode(output, np.ones((3, 3)), iterations=1)
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('Goutput')
                # plt.show()
                output = output * (1 - cv2.dilate(image, np.ones((3, 3)), iterations=1))
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('326')
                # plt.show()
                output_mask = cv2.dilate(output, np.ones((7, 7)), iterations=1)
                # plt.imshow(output_mask[950:1150,400:600], 'gray')
                # plt.title('output_mask')
                # plt.show()
                output += image
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('334')
                # plt.show()
                # output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((7, 7)), iterations=1)
                output = cv2.dilate(output, np.ones((3, 3)), iterations=1)
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('339')
                # plt.show()
                output = cv2.blur(output, (3, 3))
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('343')
                # plt.show()
                # output = cv2.erode(output, np.ones((3, 3)), iterations=2)
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('347')
                # plt.show()
                output = output * output_mask
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('353')
                # plt.show()
                # output = cv2.dilate(output, np.ones((3, 3)), iterations=3)
                # output = cv2.blur(output, (9, 9))
                # output = cv2.erode(output, np.ones((3, 3)), iterations=2)
                # output = output * (1 - image)
                # output = cv2.erode(output, np.ones((3, 3)), iterations=1)
                # output = output * (1 - image // 255)
                # output = cv2.dilate(output, np.ones((3, 3)), iterations=1) * 255
                output[output > 1] = 1
                output[image == 1] = 1
                # plt.imshow(output[950:1150,400:600], 'gray')
                # plt.title('output')
                # plt.show()
                # num, labels, stats, centroids = cv2.connectedComponentsWithStats(output, connectivity=8)
                # for i in range(1, num):
                #     if stats[i][-1] < 100:
                #         output[labels == i] = 0
                output = output * (1 - image) + image
                image *= 255
                output *= 255
                # image = cv2.resize(image, (3466, 5196))
                # output = cv2.resize(output, (3466, 5196))
                output[output > 127] = 255
                labeled_image, img_labeled_num = convert_labels_to_rgb(image)
                labeled_output, output_labeled_num = convert_labels_to_rgb(output)
                if save_dir is not None:
                    savepath = os.path.join(save_dir, 'rw', os.path.basename(imagepath))
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath,
                                cv2.merge([cv2.bitwise_and(image, output), cv2.bitwise_and(image, output), output]))
                    savepath = savepath.replace('rw', 'rgb')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, cv2.merge([image, image, output]))
                    savepath = savepath.replace('rgb', 'pred')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, output)
                    savepath = savepath.replace('pred', 'input')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, image)
                    savepath = savepath.replace('input', 'labeled_image')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, labeled_image)
                    savepath = savepath.replace('labeled_image', 'labeled_output')
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    cv2.imwrite(savepath, labeled_output)


class Matrice:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def update(self, pred, target):
        pred = pred > 0.5  # 假设使用阈值为0.5进行二分类
        target = target > 0.5
        self.TP += ((pred == 1) & (target == 1)).sum().item()
        self.FP += ((pred == 1) & (target == 0)).sum().item()
        self.TN += ((pred == 0) & (target == 0)).sum().item()
        self.FN += ((pred == 0) & (target == 1)).sum().item()

    def compute(self):
        accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        iou = self.TP / (self.TP + self.FP + self.FN)
        dice = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        return accuracy, precision, recall, iou, dice


def convert_labels_to_rgb(image):
    """
    convert graph components labels into RGB image
    """
    labels, num = sklabel(image, connectivity=2, return_num=True)
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img, num
