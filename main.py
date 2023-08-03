import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SwinTransformerSys
from utils import parse_args, RainDataset, rgb_to_y, psnr, ssim


def test_loop(net, data_loader, num_iter):
    # mention the images which you want to track during training
    intermediate_tracked_images = ['norain-11.png', 'norain-32.png', 'norain-40.png', 'norain-48.png', 'norain-53.png', 'norain-83.png']
    # mention the iterations at which you want to track the images
    intermediate_tracked_iters = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 30000, 60000, 100000, 130000, 160000, 180000, 200000, 230000, 260000, 290000, 320000, 340000, 350000, 380000, 400000]
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.cuda(), norain.cuda()
            out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(args.save_path, args.data_name, name[0])

            # saving intermediate images
            if num_iter in intermediate_tracked_iters:
                if name[0] in intermediate_tracked_images:
                    this_psnr = round(total_psnr / count, 2)
                    img_file_name = f"{name[0].split('.')[0]}_psnr_{this_psnr}.{name[0].split('.')[1]}"
                    this_inter_save_path = f'result/intermediate_train_images/{img_file_name}'
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))                
                    Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(this_inter_save_path)

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                     .format(num_iter, 1 if args.model_file else args.num_iter,
                                             total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args.model_file else num_iter // 1000) + 1))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(args.save_path, args.data_name), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))


if __name__ == '__main__':
    args = parse_args()
    test_dataset = RainDataset(args.data_path, args.data_name, 'test', args.patch_size[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
    model = SwinTransformerSys().cuda()
    if args.model_file:
        model.load_state_dict(torch.load(args.model_file))
        save_loop(model, test_loader, 1)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
        total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
        for n_iter in train_bar:
            # progressive learning
            if n_iter == 1 or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_iter - start_iter)
                train_dataset = RainDataset(args.data_path, args.data_name, 'train', args.patch_size[i], length)
                num = 15
                a_org, b_org, _, _, _ = test_dataset.__getitem__(num)
                print(a_org.shape)    
                train_loader = iter(DataLoader(train_dataset, args.batch_size[i], True, num_workers=args.workers))
                
                i += 1
            # train
            model.train()
            rain, norain, name, h, w = next(train_loader)
            rain, norain = rain.cuda(), norain.cuda()
            out = model(rain)
            loss = F.l1_loss(out, norain)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += rain.size(0)
            total_loss += loss.item() * rain.size(0)
            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                      .format(n_iter, args.num_iter, total_loss / total_num))

            lr_scheduler.step()
            if n_iter % 1000 == 0:
                results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                save_loop(model, test_loader, n_iter)
