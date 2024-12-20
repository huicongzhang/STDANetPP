import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.mda.functions import MSDeformAttnFunction
from basicsr.utils.registry import ARCH_REGISTRY
import math
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from basicsr.archs.rvrt_arch import RSTBWithInputConv,Upsample
from .arch_util import flow_warp, make_layer,ConvResidualBlocks



@ARCH_REGISTRY.register()
class STDANetPP(nn.Module):
    def __init__(self,
                 mid_channels=128,
                 num_blocks=5,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 cpu_cache_length=180):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.stride = 4
        self.max_keysframe = 5  
        
        # optical flow
        self.spynet = SpyNet(spynet_path)
        
        
        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(Rearrange('n d c h w -> n c d h w'),
                                              nn.Conv3d(3, 112, (1, 3, 3),
                                                        (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              nn.Conv3d(112, mid_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              Rearrange('n c d h w -> n d c h w'),
                                              RSTBWithInputConv(
                                                                in_channels=mid_channels,
                                                                kernel_size=(1, 3, 3),
                                                                groups=1,
                                                                num_blocks=1,
                                                                dim=mid_channels,
                                                                input_resolution=[1, 64, 64],
                                                                depth=1,
                                                                num_heads=4,
                                                                window_size=[1, 8, 8],
                                                                mlp_ratio=2,
                                                                qkv_bias=True, qk_scale=None,
                                                                norm_layer=nn.LayerNorm,
                                                                use_checkpoint_attn=[False],
                                                                use_checkpoint_ffn=[False]
                                                               )
            )

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.ltff = nn.ModuleDict()
        self.backbone_res = nn.ModuleDict()
        
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = DeformableAttnBlocks(n_heads=6,n_points=8,d_model=mid_channels)
                self.ltff[module] = LTFF(stride=self.stride,mid_channels=mid_channels)
            
            self.backbone_res[module] = RSTBWithInputConv(
                                                     in_channels= (3 + i) * mid_channels,
                                                     kernel_size=(1, 3, 3),
                                                     groups=2,
                                                     num_blocks=2,
                                                     dim=mid_channels,
                                                     input_resolution=[1, 64, 64],
                                                     depth=1,
                                                     num_heads=4,
                                                     window_size=[1, 8, 8],
                                                     mlp_ratio=2,
                                                     qkv_bias=True, qk_scale=None,
                                                     norm_layer=nn.LayerNorm,
                                                     use_checkpoint_attn=[False],
                                                     use_checkpoint_ffn=[False]
                                                     )
        # upsampling module
        
        self.reconstruction = RSTBWithInputConv(
                                               in_channels=5 * mid_channels,
                                               kernel_size=(1, 3, 3),
                                               groups=2,
                                               num_blocks=2,
                                               dim=mid_channels,
                                               input_resolution=[2, 64, 64],
                                               depth=2,
                                               num_heads=4,
                                               window_size=[2, 8, 8],
                                               mlp_ratio=2,
                                               qkv_bias=True, qk_scale=None,
                                               norm_layer=nn.LayerNorm,
                                               use_checkpoint_attn=[True,False],
                                               use_checkpoint_ffn=[True,False]
                                               )

        self.conv_before_upsampler = nn.Sequential(
                                                  nn.Conv3d(mid_channels, 112, kernel_size=(1, 1, 1),
                                                            padding=(0, 0, 0)),
                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                                  )
        self.upsampler = Upsample(4, 112)
        self.conv_last = nn.Conv3d(112, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name,flip_flows):
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        # init location map
        
        flow_downsample_buffers = flows.new_zeros(n, 2, h//self.stride, w//self.stride)
        # init buffer
        
        sparse_feat_buffers_s1 = []
        
        keyframe_idx = list(range(0, t+1, 3))
        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            keyframe_idx = list(range(t, 0, -3))
            
        
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        feat_LTFF = flows.new_zeros(n, self.mid_channels, h, w)
        
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
                feat_LTFF = feat_LTFF.cuda()
                
                flow_downsample_buffers = flow_downsample_buffers.cuda()
                
            
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                flip_flow_n1 = flip_flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()
                    flip_flow_n1 = flip_flow_n1.cuda()

                
                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                flip_flow_n2 = torch.zeros_like(flip_flow_n1)
                
                # update the location map
                flow_downsample = F.adaptive_avg_pool2d(flow_n1,(h//self.stride,w//self.stride))/self.stride
                
                
                flow_downsample_buffers = flow_downsample.view(n,1,2,h//self.stride,w//self.stride) + flow_warp(flow_downsample_buffers.view(n,-1,h//self.stride,w//self.stride),flow_downsample.permute(0,2,3,1)).view(n,-1,2,h//self.stride,w//self.stride)
                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flip_flow_n2 = flip_flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()
                        flip_flow_n2 = flip_flow_n2.cuda()

                    



                
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                
                feat_prop_warp = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                
                feat_LTFF = self.ltff[module_name](feat_current, feat_prop_warp, sparse_feat_buffer_s1, flow_downsample_buffers)
                
                
                
                feat_prop = self.deform_align[module_name](torch.stack([feat_current,feat_prop, feat_n2],1), flow_n1, flow_n2, flip_flow_n1, flip_flow_n2)
                
                

                
                # update the location map
                if idx in keyframe_idx:
                    
                    flow_downsample_buffers = torch.cat([flow_downsample_buffers,flow_downsample_buffers.new_zeros(n, 1, 2, h//self.stride, w//self.stride)],dim=1)


            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop] + [feat_LTFF]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            # print(feat.unsqueeze(1).shape)
            feat_prop = feat_prop + self.backbone_res[module_name](feat.unsqueeze(1)).squeeze(1)
            

            feats[module_name].append(feat_prop)
            #  update buffers
            if idx in keyframe_idx:
                
                # feature tokenization *4
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)
                if len(sparse_feat_buffers_s1) > self.max_keysframe:
                    sparse_feat_buffers_s1 = sparse_feat_buffers_s1[1:]
                    flow_downsample_buffers = flow_downsample_buffers[:,1:,:,:,:]



                

            ##############################################################
            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        
        feats['spatial'] = torch.stack(feats['spatial'], 1)
        feats['backward_1'] = torch.stack(feats['backward_1'], 1)
        feats['forward_1'] = torch.stack(feats['forward_1'], 1)
        feats['backward_2'] = torch.stack(feats['backward_2'], 1)
        feats['forward_2'] = torch.stack(feats['forward_2'], 1)


        hr = torch.cat([feats[k] for k in feats], dim=2)
        # print(hr.shape)
        hr = self.reconstruction(hr)
        hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
        # hr += torch.nn.functional.interpolate(lqs, size=hr.shape[-3:], mode='trilinear', align_corners=False)
        hr += lqs
        return hr

    def forward(self, lqs):
        

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs)
            feats['spatial'] = [feats_[:,i,...] for i in range(feats_.size(1))]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                    flip_flow = flows_forward
                elif flows_forward is not None:
                    flows = flows_forward
                    flip_flow = flows_backward
                else:
                    flows = flows_backward.flip(1)
                    flip_flow = flows_backward

                feats = self.propagate(feats, flows, module, flip_flow)
                
                
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)
class LTFF(nn.Module):
    def __init__(self, stride=4,mid_channels=128):
        super().__init__()
        self.n_heads = 4
        self.n_points = 4
        self.stride = stride
        self.out_channels = mid_channels
        self.mid_channels = 64
        self.fusion = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1, bias=True)
        
        self.attention_weights = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2, self.mid_channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.n_heads * self.n_points , 1, 1, 0),
        )
        self.sampling_offsets = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2, self.mid_channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.n_heads * self.n_points * 2, 1, 1, 0),
        )
        self.init_offset()
    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        # init sample offset
        nn.init.constant_(self.sampling_offsets[-1].weight, 0)
        nn.init.constant_(self.sampling_offsets[-1].bias, 0)

        

        _constant_init(self.attention_weights[-1],val=0, bias=0)
        # _constant_init(self.patch_offsets[-1],val=0, bias=0)

    def forward(self, curr_feat, anchor_feat, sparse_feat_set_s1, flow_buffers):
        

        n, c, h, w = anchor_feat.size()
        t = sparse_feat_set_s1.size(1)
        # print(t)
        feat_len = int(c*self.stride*self.stride)
        feat_num = int((h//self.stride) * (w//self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1] n,t,h//4,w//4,2
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        location_reference = torch.stack([grid_x,grid_y],dim=0).type_as(flow_buffers).expand(n,-1,-1,-1).view(n,1,2,h//self.stride,w//self.stride)
        location_reference.requires_grad = False
        grid_flow_res = flow_buffers.contiguous().view(n,t,2,h//self.stride,w//self.stride)
        location_feat = flow_buffers + location_reference
        grid_flow = location_feat.contiguous().view(n,t,2,h//self.stride,w//self.stride).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w//self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h//self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        
        output_s1_warp = F.grid_sample(sparse_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)

        # deformable offsets

        # (n*t, c*4*4, h//4*w//4) ----> (n*t, c, h,w) ----- > (n*t, c, h//4,w//4)
        output_s1_res = F.fold(output_s1_warp.view(n*t,c*self.stride*self.stride,(h//self.stride)*(w//self.stride)),output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        output_s1_ori = F.fold(sparse_feat_set_s1.view(n*t,c*self.stride*self.stride,(h//self.stride)*(w//self.stride)),output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        
        output_s1_extra_feat = F.adaptive_avg_pool2d(output_s1_res,(h//self.stride,w//self.stride))
        index_output_s1_extra_feat = F.adaptive_avg_pool2d(output_s1_ori,(h//self.stride,w//self.stride))
        
        extra_feat = torch.cat([index_output_s1_extra_feat,output_s1_extra_feat,grid_flow_res.contiguous().view(n*t,2,h//self.stride,w//self.stride)],dim=1)
        # (nt) * (heads*point)* 2 * (h//4) * (w//4)
        sampling_offsets = self.sampling_offsets(extra_feat).view(n*t,self.n_heads*self.n_points,2,(h//self.stride),(w//self.stride))
        
        # get deformable attention weights
        extra_feat_attn = torch.cat([index_output_s1_extra_feat,output_s1_extra_feat,grid_flow_res.contiguous().view(n*t,2,h//self.stride,w//self.stride)],dim=1)
        # n,t,n_heads,self.n_points,h//4,w//4 ---->  n,n_heads,h//4,w//4,t,points ---> n*n_heads,h//4,w//4,t*points
        attention_weights = self.attention_weights(extra_feat_attn).view(n,t,self.n_heads*self.n_points,h//self.stride,w//self.stride).view(n,t,self.n_heads,self.n_points,h//self.stride,w//self.stride).permute(0,2,4,5,1,3).contiguous().view(n*self.n_heads,(h//self.stride),(w//self.stride),t*self.n_points)
        attention_weights = torch.softmax(attention_weights,-1)
        # n*n_heads,1,h//4,w//4,t*points ---> n*n_heads,c//heads*4*4,h//4,w//4,t*points
        attention_weights = attention_weights.unsqueeze(1).expand(-1,(c//self.n_heads)*self.stride*self.stride,-1,-1,-1)
        
        output_grid_offsets  = location_feat.contiguous().view(n,t,2,h//self.stride,w//self.stride).view(n*t,1,2,h//self.stride,w//self.stride) \
             + sampling_offsets
        # (nt) * (heads*point)* 2 * (h//4) * (w//4) --> (nt) * (heads*point) * (h//4) * (w//4) * 2
        output_grid_offsets = output_grid_offsets.permute(0,1,3,4,2)
        sampling_offsets_x = 2*output_grid_offsets[:, :, :, :, 0]/(w//self.stride) - 1 
        sampling_offsets_y = 2*output_grid_offsets[:, :, :, :, 1]/(h//self.stride) - 1
        # (n) * (t) * (heads) * (h//4) * (w//4) * (point)*(2)
        output_grid_offsets =  torch.stack((sampling_offsets_x, sampling_offsets_y), dim=4)
        # (n*t*heads)  * (h//4*w//4) * (point)*(2)
        output_grid_offsets = output_grid_offsets.view(n*t,self.n_heads,self.n_points,(h//self.stride)*(w//self.stride),2).view(-1,self.n_points,(h//self.stride)*(w//self.stride),2)

        
        sparse_feat_set_s1 = F.unfold(sparse_feat_set_s1.view(-1,c*self.stride*self.stride,(h//self.stride),(w//self.stride)), kernel_size=(1,1), padding=0, stride=1) 
        output_s1 = F.fold(sparse_feat_set_s1, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        # index_output_s1 = F.fold(index_feat_set_s1.view(-1,c*self.stride*self.stride,(h//self.stride)*(w//self.stride)), output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        
        # (n*t, c, h, w) ---> (n*t*heads, c//heads, h, w)
        output_s1 = output_s1.view(n*t,self.n_heads,c//self.n_heads,h,w)
        output_s1 = output_s1.view(n*t*self.n_heads,c//self.n_heads,h,w)
        # print(output_s1.shape)

        
        output_s1 = F.unfold(output_s1, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
        output_s1 = F.fold(output_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
        # index_output_s1 = F.unfold(index_output_s1, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
    
        # (ntheads) * (c//heads*4*4) * (h//4*w//4) * points  doing
        output_s1 = output_s1.view(n,t,self.n_heads,((c//self.n_heads)*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # n,t,self.n_heads,(h//self.stride)*(w//self.stride,self.n_points,2
        output_grid_offsets = output_grid_offsets.view(n,t,self.n_heads,self.n_points,(h//self.stride)*(w//self.stride),2).permute(0,1,2,4,3,5).contiguous()
        output_s1 = F.grid_sample(output_s1.view(-1,((c//self.n_heads)*self.stride*self.stride),(h//self.stride),(w//self.stride)),output_grid_offsets.view(-1,(h//self.stride)*(w//self.stride),self.n_points,2),mode='nearest',padding_mode='zeros',align_corners=True)
        output_s1 = output_s1.view(n,t,self.n_heads,((c//self.n_heads)*self.stride*self.stride),(h//self.stride),(w//self.stride),self.n_points).permute(0,2,3,4,5,1,6).contiguous().view(n*self.n_heads,((c//self.n_heads)*self.stride*self.stride),(h//self.stride),(w//self.stride),t*self.n_points)
        
        out = output_s1 * attention_weights
        # n*n_heads,(c//heads)*4*4,h//4,w//4,t*points ---> n*n_heads,(c//heads)*4*4,h//4,w//4 ---> n*n_heads,(c//heads),h,w
        out = out.sum(-1).view(n*self.n_heads,(c//self.n_heads)*self.stride*self.stride,(h//self.stride),(w//self.stride))
        # n*n_heads,(c//heads)*4*4,h//self.stride*w//self.stride ---> n*n_heads,(c//heads),h,w ---> n,c,h,w
        out =  F.fold(out.view(n*self.n_heads,(c//self.n_heads)*self.stride*self.stride,(h//self.stride)*(w//self.stride)),output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        out = out.view(n,c,h,w)


        
        # ( n*heads) * (c//heads) * h * w ---> n * c * h * w
        out = self.fusion(out)
        out += anchor_feat
        return out

class DeformableAttnBlocks(nn.Module):
    def __init__(self,n_heads=8,n_levels=3,n_points=12,d_model=64):
        super(DeformableAttnBlocks, self).__init__()
        # self.MMA = DeformableAttnBlock_PDA(n_heads,n_levels,n_points,d_model)
        self.pad_layer1 = DeformableAttnBlock_PDA(n_heads,n_levels,n_points,d_model)
        # self.pad_layer2 = DeformableAttnBlock_PDA(n_heads,n_levels,n_points,d_model)
    def forward(self, x, flow_1, flow_2, flip_flow_1,flip_flow_2):
        # x = self.MMA(x, flow_1, flow_2, flip_flow_1,flip_flow_2)
        """  out = self.MSA1(x, flow_1, flow_2)
        out = self.MSA2(torch.stack([out,x[:,1],x[:,2]],1), flow_1, flow_2) """
        if self.training == True:
            out = checkpoint(self.pad_layer1,x, flow_1, flow_2)
        else:
            out = self.pad_layer1(x, flow_1, flow_2)
            
        return out
class DeformableAttnBlock_PDA(nn.Module):
    def __init__(self,n_heads=8,n_levels=3,n_points=12,d_model=64):
        super(DeformableAttnBlock_PDA, self).__init__()
        
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.out_channels = d_model
        self.mid_channels = 64
        self.patch_size = 3
        self.attention_weights = nn.Sequential(
            nn.Conv2d(n_levels * self.out_channels + 4, self.mid_channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, n_heads * n_levels * n_points * self.patch_size * self.patch_size, 1, 1, 0),
        )
        self.sampling_offsets = nn.Sequential(
            nn.Conv2d(n_levels * self.out_channels + 4, self.mid_channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, n_heads * n_levels * n_points * 2, 1, 1, 0),
        )
        self.patch_offsets = nn.Sequential(
            nn.Conv2d(n_levels * self.out_channels + 4, self.mid_channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, n_heads * n_levels * n_points, 1, 1, 0),
        )
        
        self.value_proj = nn.Sequential(
            nn.Conv2d(n_levels * self.out_channels,n_levels * self.out_channels, 1, 1, 0),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
        )

        
        
        self.init_offset()
    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        # init sample offset
        nn.init.constant_(self.sampling_offsets[-1].weight, 0)
        nn.init.constant_(self.sampling_offsets[-1].bias, 0)


        _constant_init(self.attention_weights[-1],val=0, bias=0)
        _constant_init(self.patch_offsets[-1],val=0, bias=0)
        
    def flow_guid_offset(self,flow_cn1, flow_cn2, offset):
        N,HW,n_heads,n_levels,n_points,_ = offset.shape
        n,_,h,w = flow_cn1.shape
        flow_0 = torch.zeros_like(flow_cn1).reshape(N,HW,1,1,1,2).repeat(1,1,n_heads,1,n_points,1)
        flow_1_flat = flow_cn1.reshape(N,2,HW).transpose(1, 2).contiguous().reshape(N,HW,1,1,1,2).repeat(1,1,n_heads,1,n_points,1)
        flow_2_flat = flow_cn2.reshape(N,2,HW).transpose(1, 2).contiguous().reshape(N,HW,1,1,1,2).repeat(1,1,n_heads,1,n_points,1)
        flow_stack = torch.cat([flow_0,flow_1_flat,flow_2_flat],3)
        
        offset_out = offset + flow_stack
        return offset_out.contiguous()
    


        
    def get_point_patch_n(self,dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.patch_size-1)//2, (self.patch_size-1)//2+1),
            torch.arange(-(self.patch_size-1)//2, (self.patch_size-1)//2+1))
        # (patch_size,patch_size,2)
        p_n = torch.stack([p_n_x, p_n_y], -1).view(self.patch_size*self.patch_size,2).type(dtype)*1
        return p_n
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        
        return reference_points
    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def preprocess(self,srcs):
        bs,t,c,h,w = srcs.shape
        masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(t):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes,valid_ratios
    
    def forward(self, x, flow_1, flow_2):
        # flow n,t-1,2,h,w
        n,t,c,h,w = x.shape
        
        
        flow_cn1 = flow_1
        flow_cn2 = flow_1 + flow_warp(flow_2,flow_1.permute(0, 2, 3, 1))
        
        warp_featcn1 = flow_warp(x[:,1],flow_cn1.permute(0, 2, 3, 1))
        warp_featcn2 = flow_warp(x[:,2],flow_cn2.permute(0, 2, 3, 1))

        extra_feat = torch.cat([x[:,0], warp_featcn1, warp_featcn2,  flow_cn1, flow_cn2], dim=1)
        
        value = self.value_proj(x.reshape(n,t*c,h,w)).reshape(n,t,c,h,w)
        # value = torch.stack(x,dim=1)
        sampling_offsets = self.sampling_offsets(extra_feat)
        attention_weights = self.attention_weights(extra_feat)
        


        spatial_shapes,valid_ratios = self.preprocess(value)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes[0].reshape(1,2),valid_ratios,device=value.device)

        # reshape
        # n,t,c,h,w--->n,t,c,h*w---->n,t,h*w,c----->n,t*h*w,c
        value = value.reshape(n,t,c,h*w).permute(0, 1, 3, 2).reshape(n,t*h*w,c)
        N, Len_in, _ = value.shape
        value = value.reshape(N, Len_in, self.n_heads, self.out_channels// self.n_heads)
        # n,n_levels * n_heads * n_levels * n_points*2,h,w ---> n,t*h*w, self.n_heads, self.n_levels, self.n_points, 2
        sampling_offsets = sampling_offsets.reshape(n,-1,h*w).permute(0, 2, 1)
        sampling_offsets = sampling_offsets.reshape(n,h*w, self.n_heads, self.n_levels, self.n_points, 2)
        
        patch_offsets = self.patch_offsets(extra_feat)
        patch_offsets = patch_offsets.reshape(n,-1,h*w).permute(0, 2, 1)
        patch_offsets = patch_offsets.reshape(n,h*w, self.n_heads, self.n_levels, self.n_points)
        patch_offsets = torch.sigmoid(patch_offsets)*3
        
        #  n,n_levels * n_heads * n_levels * n_points*2,h,w ---> n,t*h*w, self.n_heads, self.n_levels, self.n_points
        attention_weights = attention_weights.reshape(n,-1,h*w).permute(0, 2, 1)
        
        attention_weights = attention_weights.reshape(n,h*w, self.n_heads, self.n_levels*self.n_points*self.patch_size*self.patch_size)
        attention_weights = F.softmax(attention_weights,-1).reshape(n,h*w, self.n_heads, self.n_levels, self.n_points*self.patch_size*self.patch_size)
        
        
        sampling_offsets = self.flow_guid_offset(flow_cn1, flow_cn2, sampling_offsets)


        # n,h*w, self.n_heads, self.n_levels, self.n_points, 2
        # init patch offsets (patch_size,patch_size,2)
        patch_point_offsets = self.get_point_patch_n(sampling_offsets.data.type()).view(1, 1, 1, 1, 1, self.patch_size*self.patch_size, 2) 

        patch_point_offsets = patch_point_offsets.repeat(n, h*w, self.n_heads, self.n_levels, self.n_points,1,1) \
                                * patch_offsets.view(n, h*w, self.n_heads, self.n_levels, self.n_points, 1, 1).repeat(1,1,1,1,1,self.patch_size*self.patch_size,2)
        sampling_offsets = patch_point_offsets \
                            + sampling_offsets.view(n, h*w, self.n_heads, self.n_levels, self.n_points, 1, 2).repeat(1,1,1,1,1,self.patch_size*self.patch_size,1)
        
                           
        sampling_offsets = sampling_offsets.reshape(n,h*w, self.n_heads, self.n_levels, self.n_points*self.patch_size*self.patch_size, 2)
        
        
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        # print(reference_points[:, :, None, :, None, :].shape)
        # print(offset_normalizer[None, None, None, :, None, :].shape)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        output = MSDeformAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 64)
        # print(output.shape)
        output = self.out_proj(output.reshape(n,h,w,c).permute(0, 3, 1, 2)) + x[:,0]
        
        return output

class DeformableAttnBlock_MMA(nn.Module):
    def __init__(self,n_heads=8,n_levels=3,n_points=12,d_model=64):
        super(DeformableAttnBlock_MMA, self).__init__()
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.out_channels = d_model
        self.patch_size = 3
        self.attention_weights = nn.Sequential(
            nn.Conv2d(7 * self.out_channels + 6*2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, n_levels*n_heads * n_levels * n_points * 9, 3, 1, 1),
        )
        self.sampling_offsets = nn.Sequential(
            nn.Conv2d(7 * self.out_channels + 6*2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, n_levels * n_heads * n_levels * n_points*2, 3, 1, 1),
        )
        
        self.value_proj = nn.Sequential(
            nn.Conv2d(n_levels*self.out_channels, n_levels*self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_levels*self.out_channels, n_levels*self.out_channels, 3, 1, 1),
        )
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(n_levels*self.out_channels, n_levels*self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_levels*self.out_channels, n_levels*self.out_channels, 3, 1, 1),
        )
        self.patch_point_offsets = self.get_point_patch_n()
    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        

        _constant_init(self.attention_weights[-1],val=0, bias=0)

    def flow_guid_offset(self,flow_cn1, flow_cn2, flow_n1c, flow_n1n2, flow_n2n1, flow_n2c, offset):
        # N, h*w, self.n_heads, self.n_levels, self.n_points, 2
        # N,HW,n_heads,n_levels,n_points,_ = offset.shape

        
        
        n, THW, n_heads, n_levels, n_points, _ = offset.shape
        t = self.n_levels
        _,_,h,w =flow_cn1.shape
        offset = offset.reshape(n, n_levels, -1, n_heads, n_levels, n_points, 2)
        offset_cer,offset_n1,offset_n2 = torch.chunk(offset, n_levels, dim=1)
        # offset_o0: n, 1, hw, n_heads, n_levels, n_points, 2
        # flow_backwards: n,t-1,2,h,w
        # cer
        

        flow_zero = torch.zeros_like(flow_n2c)
        offset_cer = offset_cer + torch.stack([flow_zero,flow_cn1,flow_cn2],1).permute(0,3,4,1,2).view(n,h*w,n_levels,2)[:,None,:,None,:,None,:]
        offset_n1 = offset_n1 + torch.stack([flow_n1c,flow_zero,flow_n1n2],1).permute(0,3,4,1,2).view(n,h*w,n_levels,2)[:,None,:,None,:,None,:]
        offset_n2 = offset_n2 + torch.stack([flow_n2c,flow_n2n1,flow_zero],1).permute(0,3,4,1,2).view(n,h*w,n_levels,2)[:,None,:,None,:,None,:]
        

        offset_out = torch.cat([offset_cer,offset_n1,offset_n2],dim=1).view(n,t*h*w,n_heads,n_levels,n_points,2)


        
        
            
            


        return offset_out
    


        
    def get_point_patch_n(self,dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.patch_size-1)//2, (self.patch_size-1)//2+1),
            torch.arange(-(self.patch_size-1)//2, (self.patch_size-1)//2+1))
        # (patch_size,patch_size,2)
        p_n = torch.stack([p_n_x, p_n_y], -1).view(self.patch_size*self.patch_size,2).type(dtype)
        return p_n
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def preprocess(self,srcs):
        bs,t,c,h,w = srcs.shape
        masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(t):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes,valid_ratios
    def forward(self, x, flow_1, flow_2, flip_flow_1,flip_flow_2):
        # flow n,t-1,2,h,w
        n,t,c,h,w = x.shape
        
        # print("flow_mean:{},std:{}".format(flow_1.mean(),flow_1.std()))
        # print("input_mean:{},std:{}".format(x[0].mean(),x[0].std()))
        flow_cn1 = flow_1
        flow_cn2 = flow_1 + flow_warp(flow_2,flow_1.permute(0, 2, 3, 1))
        
        warp_featcn1 = flow_warp(x[:,1],flow_cn1.permute(0, 2, 3, 1))
        warp_featcn2 = flow_warp(x[:,2],flow_cn2.permute(0, 2, 3, 1))


        # n1
        flow_n1c = flip_flow_1
        flow_n1n2 = flow_2
        warp_featn1c = flow_warp(x[:,0],flow_n1c.permute(0, 2, 3, 1))
        warp_featn1n2 = flow_warp(x[:,2],flow_n1n2.permute(0, 2, 3, 1))

        # n2
        flow_n2n1 = flip_flow_2
        flow_n2c = flip_flow_2 + flow_warp(flip_flow_1,flip_flow_2.permute(0, 2, 3, 1))
        warp_n2n1 = flow_warp(x[:,1],flow_n2n1.permute(0, 2, 3, 1))
        warp_n2c = flow_warp(x[:,0],flow_n2c.permute(0, 2, 3, 1))


        extra_feat = torch.cat([x[:,0], warp_featcn1, warp_featcn2, warp_featn1c, warp_featn1n2, warp_n2n1, warp_n2c, flow_cn1, flow_cn2, flow_n1c, flow_n1n2, flow_n2n1, flow_n2c], dim=1)
        # out = self.conv_offset(extra_feat)
        # out = self.sampling_offsets(extra_feat)
        value = self.value_proj(x.reshape(n,t*c,h,w)).reshape(n,t,c,h,w)
        # value = torch.stack(x,dim=1)
        sampling_offsets = self.sampling_offsets(extra_feat)
        attention_weights = self.attention_weights(extra_feat)

        spatial_shapes,valid_ratios = self.preprocess(value)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes,valid_ratios,device=value.device)

        # reshape
        # n,t,c,h,w--->n,t,c,h*w---->n,t,h*w,c----->n,t*h*w,c
        value = value.reshape(n,t,c,h*w).permute(0, 1, 3, 2).reshape(n,t*h*w,c)
        N, Len_in, _ = value.shape
        value = value.reshape(N, Len_in, self.n_heads, self.out_channels// self.n_heads)
        # n,n_levels * n_heads * n_levels * n_points*2,h,w ---> n,t*h*w, self.n_heads, self.n_levels, self.n_points, 2
        sampling_offsets = sampling_offsets.reshape(n,t,-1,h,w).reshape(n,t,-1,h*w).permute(0, 1, 3, 2).reshape(n,t*h*w,-1)
        sampling_offsets = sampling_offsets.reshape(n,t*h*w, self.n_heads, self.n_levels, self.n_points, 2)

        #  n,n_levels * n_heads * n_levels * n_points*2,h,w ---> n,t*h*w, self.n_heads, self.n_levels, self.n_points
        attention_weights = attention_weights.reshape(n,t,-1,h,w).reshape(n,t,-1,h*w).permute(0, 1, 3, 2).reshape(n,t*h*w,-1)
        attention_weights = attention_weights.reshape(n,t*h*w, self.n_heads, self.n_levels*self.n_points*9)
        # attention_weights = F.sigmoid(attention_weights,-1)
        attention_weights = F.softmax(attention_weights,-1).reshape(n,t*h*w, self.n_heads, self.n_levels,self.n_points*9)
        # init patch attention
        
        


        
        #  flow guide
        sampling_offsets = self.flow_guid_offset(flow_cn1, flow_cn2, flow_n1c, flow_n1n2, flow_n2n1, flow_n2c, sampling_offsets)
        # n,t*h*w, self.n_heads, self.n_levels, self.n_points, 2
        # init patch offsets (patch_size,patch_size,2)
        patch_point_offsets = self.get_point_patch_n(sampling_offsets.data.type()).view(1, 1, 1, 1, 1, self.patch_size*self.patch_size, 2) 
        sampling_offsets = patch_point_offsets \
                            + sampling_offsets.reshape(n, t*h*w, self.n_heads, self.n_levels, self.n_points, 1, 2)
        sampling_offsets = sampling_offsets.reshape(n, t*h*w, self.n_heads, self.n_levels, self.n_points*self.patch_size*self.patch_size, 2)

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        
        # print(offset_normalizer[None, None, None, :, None, :].shape)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = MSDeformAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 64)
        output = output.reshape(n,t,h,w,c).permute(0, 1, 4, 2, 3)
        output = self.out_proj(output.reshape(n,t*c,h,w)).reshape(n,t,c,h,w) + x

        return output

