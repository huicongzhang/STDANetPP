# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os

import importlib
import torch
# import os
import torchvision
from torch import distributed as dist
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.utils import get_root_logger, imwrite, tensor2img
from importlib import import_module
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
import basicsr.losses as loss
from skimage.metrics import structural_similarity as SSIM_
from skimage.metrics import peak_signal_noise_ratio as PSNR_
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.logger import AverageMeter
from basicsr.archs.arch_util import flow_warp
import time
def create_video_model(opt):
    module = import_module('basicsr.archs.' + opt['model'].lower())
    model = module.make_model(opt)
    return model

metric_module = importlib.import_module('basicsr.metrics')

def generate_kernels(h=11,l=80,n=10):
    kernels = torch.zeros(l,1,h,h).to(torch.device('cuda'))
    n2 = (n-1)//2
    n1 = n-2*n2
    kernels[0*n2:1*n2,:,0,0] = 1
    kernels[1*n2:2*n2,:,0,h//4] = 1
    kernels[2*n2:3*n2,:,0,h//2] = 1
    kernels[3*n2:4*n2,:,0,3*h//4] = 1
    kernels[4*n2:5*n2,:,0,h-1] = 1
    kernels[5*n2:6*n2,:,h-1,0] = 1
    kernels[6*n2:7*n2,:,h-1,h//4] = 1
    kernels[7*n2:8*n2,:,h-1,h//2] = 1
    kernels[8*n2:9*n2,:,h-1,3*h//4] = 1
    kernels[9*n2:10*n2,:,h-1,h-1] = 1
    kernels[10*n2:11*n2,:,h//4,0] = 1
    kernels[11*n2:12*n2,:,h//4,h-1] = 1
    kernels[12*n2:13*n2,:,h//2,0] = 1
    kernels[13*n2:14*n2,:,h//2,h-1] = 1
    kernels[14*n2:15*n2,:,3*h//4,0] = 1
    kernels[15*n2:16*n2,:,3*h//4,h-1] = 1
    kernels[16*n2+0*n1:16*n2+1*n1,:,h//4,h//4] = 1
    kernels[16*n2+1*n1:16*n2+2*n1,:,h//4,h//2] = 1
    kernels[16*n2+2*n1:16*n2+3*n1,:,h//4,3*h//4] = 1 
    kernels[16*n2+3*n1:16*n2+4*n1,:,h//2,h//4] = 1
    kernels[16*n2+4*n1:16*n2+5*n1,:,h//2,3*h//4] = 1
    kernels[16*n2+5*n1:16*n2+6*n1,:,3*h//4,h//4] = 1
    kernels[16*n2+6*n1:16*n2+7*n1,:,3*h//4,h//2] = 1
    kernels[16*n2+7*n1:16*n2+8*n1,:,3*h//4,3*h//4] = 1
    return kernels

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

@MODEL_REGISTRY.register()
class ModelVRT(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ModelVRT, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')
        
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model10_to_device(self.net_g)
        # self.n_sequence = opt['n_sequence']
        print("define network cdvd_tsp")
        self.print_network(self.net_g)
        

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        # self.loss = loss.Loss2(opt['loss_type'])
        self.scaler = torch.cuda.amp.GradScaler()
        self.no_fix_flow = self.opt["no_fix_flow"]
        self.have_fix_flow = False
    
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.log_dict = OrderedDict()
        # define losses
        if train_opt.get('pixel_opt'):
            # pixel_type = train_opt['pixel_opt'].pop('type')
            # cri_pix_cls = getattr(loss_module, pixel_type)
            # self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                # self.device)
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.log_dict['l_pix'] = AverageMeter()
            # self.log_dict['ori_l_pix'] = AverageMeter()

            
            # self.log_dict['l_fb'] = AverageMeter()
            # self.log_dict['l_f'] = AverageMeter()
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            # to do
            pass
            """ percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device) """
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    def model10_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            # find_unused_parameters = self.opt.get('find_unused_parameters',
            # False                                      )
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False
                )
            net._set_static_graph()
            #outputs, _ = net(torch.zeros(1,6,3,64,64).cuda(), self.kernel1, self.kernel2, self.kernel3)
            #outputs[0].sum().backward()
            #for n, p in net.named_parameters():
            #    if p.grad is None and p.requires_grad is True:
            #        print('Params not used', n, p.shape)
            # exit(0)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []



        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                # print(k)
                if 'spynet' in k :
                # if k.startswith('module.spynet'):
                    optim_params_lowlr.append(v)
                    print("lower lr", k)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = self.opt['ratio']

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        # elif optim_type == 'SGD':
        #     self.optimizer_g = torch.optim.SGD(optim_params,
        #                                        **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        # print(self.optimizer_g)
        # exit(0)

    def feed_data(self, data):
        lq, gt = data['lq'],data['gt']
        self.lq = lq.to(self.device)
        self.gt = gt.to(self.device)
        
    def feed_data_test(self,data):
        lq, gt = data['L'],data['H']
        self.lq = lq.to(self.device)
        self.gt = gt.to(self.device)
    def feed_data_test2(self,data):
        lq, gt = data['L'],data['H']
        self.lq = lq.to(self.device).unsqueeze(0)[:,:100,...]
        self.gt = gt.to(self.device).unsqueeze(0)[:,:100,...]
    
    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t


    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        

        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq


    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1 or self.have_fix_flow == False:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name :
                        param.requires_grad_(False)
                        logger.warning("fix parameters:{}".format(name))
                self.have_fix_flow = True
            # self.no_fix_flow == True and 
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)
        if self.no_fix_flow == True:
            logger = get_root_logger()
            logger.warning('Train all the parameters.')
            self.net_g.requires_grad_(True)
            self.no_fix_flow = False
        self.optimizer_g.zero_grad()
        #print(self.lq.shape, self.gt.shape)
        
        n,t,c,h,w = self.lq.shape
        
        
        output = self.net_g(self.lq)
        


        # print(self.lq.shape, output.shape, output_1.shape, self.gt_1.shape, self.gt.shape)
        # exit(0)
        self.output = output

        # l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        
        l_pix = self.cri_pix(output, self.gt)
        
        loss_dict['l_pix'] = l_pix
        
        
        
        l_total = l_pix

        l_total.backward()
        self.optimizer_g.step()
        
        
        for k,v in self.reduce_loss_dict(loss_dict).items():
            self.log_dict[k].update(v)
        # exit(0)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()
        """ 
        whole_t = self.lq.size(1)
        whole_out = torch.zeros_like(self.lq)
        stride = 100
        t_idx_list = list(range(0, whole_t-100, stride)) + [max(0, whole_t-100)]
        """
    def test_by_clip(self):
        self.net_g.eval()
        whole_t = self.lq.size(1)
        whole_out = torch.zeros_like(self.lq)
        stride = 100
        t_idx_list = list(range(0, whole_t-100, stride)) + [max(0, whole_t-100)]
        with torch.no_grad():
            for t_idx in t_idx_list:
                clip_lq = self.lq[:,t_idx:t_idx+100,:,:,:]
                clip_out = self.test_by_patch(clip_lq)
                whole_out[:,t_idx:t_idx+100,:,:,:] = clip_out
        self.output = whole_out
        self.net_g.train()
    def test_by_patch(self,lq):
        
        
        d_old = lq.size(1)
        d_pad = (d_old// 2+1)*2 - d_old
        pad_lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
        with torch.no_grad():
            size_patch_testing = 256
            overlap_size = 20
            b,t,c,h,w = pad_lq.shape
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
            w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
            E = torch.zeros(b, t, c, h, w)
            W = torch.zeros_like(E)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = pad_lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                    
                    out_patch = self.net_g(in_patch)
                    
                    
                    out_patch = out_patch.detach().cpu().reshape(b,t,c,size_patch_testing,size_patch_testing)

                    out_patch_mask = torch.ones_like(out_patch)

                    if True:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size//2:, :] *= 0
                            out_patch_mask[..., -overlap_size//2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size//2:] *= 0
                            out_patch_mask[..., :, -overlap_size//2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_size//2, :] *= 0
                            out_patch_mask[..., :overlap_size//2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_size//2] *= 0
                            out_patch_mask[..., :, :overlap_size//2] *= 0

                    E[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch)
                    W[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch_mask)
            output = E.div_(W)
        output = output[:, :d_old, :, :, :]
        return output 
    def get_latest_images(self):
        return [self.lq[0].float(), self.gt[0].float(), self.output[0].float()]

    def single_image_inference(self, img, save_path):
        self.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if self.opt['val'].get('grids', False):
            self.grids()

        self.test()

        if self.opt['val'].get('grids', False):
            self.grids_inverse()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)
    def validation(self, dataloader, current_iter, tb_logger,wandb_logger=None,save_img=False):
        """Validation function.
        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            wandb_loggger (wandb logger): wandb runer logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, wandb_logger, save_img,rgb2bgr=True, use_image=True)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, wandb_logger, save_img,rgb2bgr=True, use_image=True)
    def dist_validation(self, dataloader, current_iter, tb_logger,wandb_logger, save_img, rgb2bgr=True, use_image=True):
        dataset = dataloader.dataset
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            # if not hasattr(self, 'metric_results'):  # only execute in the first run
                
            self.metric_results = {
            metric: torch.zeros(1, dtype=torch.float32, device='cuda')
            for metric in self.opt['val']['metrics'].keys()
            }
            cnt = torch.zeros(1, dtype=torch.float32, device='cuda')
        rank, world_size = get_dist_info()
        num_seq = len(dataset)
        num_pad = (world_size - (num_seq % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='image')
            show_dir_name = f"results/{self.opt['name']}"
        metric_data = dict()
        for i in range(rank, num_seq + num_pad, world_size):
            idx_data = min(i,num_seq - 1)
            # print(idx_data)
            val_data = dataset[idx_data]
            filenames = val_data["folder"]
            self.feed_data_test2(val_data)
            self.test_by_patch()
            visuals = self.get_current_visuals()
            del self.lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()
            
            if i < num_seq:
                
                b,t,c,h,w = visuals['result'].shape
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img
                    if with_metrics:
                        # print(f"#####################{with_metrics}#####################")
                        # calculate metrics
                        opt_metric = deepcopy(self.opt['val']['metrics'])
                        if use_image:
                            for name, opt_ in opt_metric.items():
                                metric_type = opt_.pop('type')
                                self.metric_results[name] += getattr(
                                    metric_module, metric_type)(metric_data['img'], metric_data['img2'], **opt_) 
                    cnt += 1
                if rank == 0:
                    for _ in range(world_size):
                        
                        pbar.update(1)
                        pbar.set_description(f'Folder: {filenames}')
        if rank == 0:
            pbar.close()
        if with_metrics:
            if self.opt['dist']:
                
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.reduce(cnt,0)
                dist.barrier()

                
            if rank == 0:
                for name, opt_ in opt_metric.items():
                    self.metric_results[name] = self.metric_results[name].item()/cnt.item()
                    # print(cnt.item())
                self._log_validation_metric_values(current_iter, dataset_name,
                                            tb_logger,wandb_logger)
                
        out_metric = 0.
        for name, opt_ in opt_metric.items():
            out_metric = self.metric_results[name]
        
        return out_metric

    def nondist_validation(self, dataloader, current_iter, tb_logger,wandb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        metric_data = dict()
        show_dir_name = f"results/{self.opt['name']}"
        os.makedirs(show_dir_name, exist_ok=True)
        for idx, val_data in enumerate(dataloader):
            
            filenames = val_data["folder"]

            self.feed_data_test(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()

            # self.test()
            
            self.test_by_clip()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            # sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            # if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)

            
            del self.lq
            del self.output
            
            
            # del self.output_flows
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            
            
            # print(visuals['result'].shape)
            b,t,c,h,w = visuals['result'].shape
            for idx in range(visuals['result'].size(1)):
                result = visuals['result'][0, idx, :, :, :]
                result_img = tensor2img([result])  # uint8, bgr
                metric_data['img'] = result_img
                if 'gt' in visuals:
                    gt = visuals['gt'][0, idx, :, :, :]
                    gt_img = tensor2img([gt])  # uint8, bgr
                    metric_data['img2'] = gt_img
                if with_metrics:
                    # calculate metrics
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(metric_data['img'], metric_data['img2'], **opt_) 
                cnt += 1
                
            # show_lqs = torchvision.utils.make_grid(visuals['result'][:,:16,...].view(b*16,c,h,w),4)
            for i in range(t):
                os.makedirs(osp.join(show_dir_name,"visualization", filenames[0]), exist_ok=True)
                torchvision.utils.save_image(visuals['result'][:,i,...],osp.join(show_dir_name,"visualization", filenames[0], f'{i:06d}.png'))



            



             

            pbar.update(1)
            
            pbar.set_description(f'Test {filenames} {filenames} {self.metric_results["psnr"]/cnt}')
            
            # if cnt == 300:
            #     break
        pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger,wandb_logger)
            print(current_metric)
        return current_metric
    
            


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger,wandb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                if wandb_logger is not None:
                    wandb_logger.log({f'metrics/{metric}':value},current_iter)
                # wandb_logger.log(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
