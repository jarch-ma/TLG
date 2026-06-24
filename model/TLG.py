import pdb
from functools import reduce
from operator import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cudnn import benchmark
from torchvision.models import resnet
from torchvision.models import vgg
# from .base.feature import extract_feat_vgg, extract_feat_res
from model.base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
from model.tlg_module import zeroshot_classifier
from generate_cam_voc import PASCAL_CLASSES
from generate_cam_coco import COCO_CLASSES
from model.clip_text import voc_class_names,refine_voc_names, coco_class_names, refine_coco_names, voc_back_class, coco_back_class
from .attention import Attention, CrossAttention
import ot
import numpy as np
import clip


class tlg(nn.Module):

    def __init__(self, backbone, use_original_imgsize, benchmark, clip_model):
        super(tlg, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=False)
            # Please replace this with your file path.
            ckpt = torch.load('/your_data_path/tlg_pretrain/vgg16-397923af.pth',
                              weights_only=True)
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]

        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=False)
            # Please replace this with your file path.
            ckpt = torch.load('/your_data_path/tlg_pretrain/resnet50-19c8e357.pth',
                              weights_only=True)
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.conv1024_512 = nn.Conv2d(1024, 512, kernel_size=1)

        #  clip
        if benchmark == 'pascal':
            self.clip_fg_embed = zeroshot_classifier(voc_class_names, ['a clean origami {}.'], clip_model)
            self.clip_refine = zeroshot_classifier(refine_voc_names, ['a clean origami {}.'], clip_model)
            self.clip_bg_embed = zeroshot_classifier(voc_back_class, ['a clean origami {}.'], clip_model)
        elif benchmark == 'coco':
            self.clip_fg_embed = zeroshot_classifier(coco_class_names, ['a clean origami {}.'], clip_model)
            self.clip_refine = zeroshot_classifier(refine_coco_names, ['a clean origami {}.'], clip_model)
            self.clip_bg_embed = zeroshot_classifier(coco_back_class, ['a clean origami {}.'], clip_model)
        else:
            raise Exception('Unavailable backbone: %s' % backbone)


        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))

        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]

        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()


        outch1, outch2, outch3 = 16, 64, 128
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

        self.res = nn.Sequential(nn.Conv2d(3, 10, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(10, 2, kernel_size=1))

        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.linear_512_625 = nn.Linear(512, 625)

        # change ch with different layers
        self.conv_512_128 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_1024_128 = nn.Conv2d(1024, 128, kernel_size=1)
        self.conv_2048_128 = nn.Conv2d(2048, 128, kernel_size=1)
        self.conv_2_1 = nn.Conv2d(2, 1, kernel_size=1)
        self.voc_conv_bach =nn.Conv2d(25,8, kernel_size=1)
        self.coco_conv_batch = nn.Conv2d(23, 8, kernel_size=1)

        self.merge = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=1, padding=0, bias=False)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.proj = nn.Linear(50, 50, bias=False)

        self.SelfAttention = Attention(dim=128, num_heads=8)
        self.CrossAttention = CrossAttention(dim=128, attn_drop=0., proj_drop=0.1)
        self.adapt_pool = nn.AdaptiveAvgPool1d(50)
        self.proj = nn.Linear(50, 2500, bias=False)
        self.final_cat = nn.Conv2d(384,128,kernel_size=1)

    def forward(self, query_img, support_img, support_cam, query_cam,
                query_mask=None, support_mask=None, class_id=None):
        with torch.no_grad():
            # backbone
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            # extracting feature
            if len(query_feats) == 7:  # VGG Backbone
                isvgg = True

                # Extract features from different layers
                support_low = support_feats[2]
                support_mid = support_feats[5]
                support_hig = support_feats[6]

                query_low = query_feats[0]
                query_mid = query_feats[3]
                query_hig = query_feats[6]

                s_local_res = support_mid
                q_local_res = support_mid

            else:
                isvgg = False  # ResNet-50 Backbone

                # 0-3 low layer：     torch.Size([bs, 512, 50, 50])
                # 4-9 middle layer：  torch.Size([bs, 1024, 25, 25])
                # 10-12 high layer：  torch.Size([bs, 2048, 13, 13])

                # Extract Cross layer
                support_low = support_feats[3]
                support_mid = support_feats[9]
                support_hig = support_feats[12]

                query_low = query_feats[0]
                query_mid = query_feats[4]
                query_hig = query_feats[10]

                s_local_res = support_mid
                q_local_res = support_mid


            # Embedding clip text features
            
            clip_fg_embed = self.clip_fg_embed.unsqueeze(1).to(query_img.device)[class_id]
            clip_fg_embed = self.linear_512_625(clip_fg_embed.float())  # torch.Size([20, 625])

            clip_refine_embed = self.clip_refine.unsqueeze(1).to(query_img.device)[class_id]
            clip_refine_embed = self.linear_512_625(clip_refine_embed.float())

            clip_bg_embed = self.clip_bg_embed.unsqueeze(1).to(query_img.device)[class_id]
            clip_bg_embed = self.linear_512_625(clip_bg_embed.float()).to(query_img.device)
            
            batch, ch, _= clip_fg_embed.size()

            clip_refine_embed = clip_refine_embed.view(batch, ch, 25, 25)
            clip_fg_embed = clip_fg_embed.view(batch, ch, 25, 25)
            clip_bg_embed = clip_bg_embed.view(batch, ch, 25, 25)

            clip_merge_embed = torch.cat([clip_fg_embed, clip_bg_embed], dim=1)
            clip_merge_embed = self.conv_2_1(clip_merge_embed)
            
        q_low_feat, q_mid_feat, q_hig_feat = self.reshape_features(query_low, query_mid, query_hig)
        s_low_feat, s_mid_feat, s_hig_feat = self.reshape_features(support_low, support_mid, support_hig)


        support_feat = torch.cat([s_low_feat, s_mid_feat, s_hig_feat], dim=1)
        query_feat = torch.cat([q_low_feat, q_mid_feat, q_hig_feat], dim=1)

        support_feat = self.final_cat(support_feat)
        query_feat = self.final_cat(query_feat)

        bsz,ch,h,w = support_feat.size()

        corr_query = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
        corr_support = Correlation.multilayer_correlation(support_feats, query_feats, self.stack_ids)

        after4d_query = self.hpn_learner.forward_conv4d(corr_query)
        after4d_support = self.hpn_learner.forward_conv4d(corr_support)

        support_feat = self.merge(torch.cat([support_feat, after4d_query], 1))
        query_feat = self.merge(torch.cat([query_feat, after4d_support], 1))

        s_attn = support_feat.view(bsz, ch, -1)
        s_attn = self.adapt_pool(s_attn).permute(0, 2, 1)  # B ,H*W, C

        q_attn = query_feat.view(bsz, ch, -1)
        q_attn = self.adapt_pool(q_attn).permute(0, 2, 1)  # B ,H*W, C

        support_attn = self.SelfAttention(s_attn)
        support_attn = self.CrossAttention(support_attn, s_attn, s_attn, OT=True).permute(0, 2, 1)
        support_attention = self.proj(support_attn)
        support_attention = support_attention.view(bsz, ch, h, w)

        query_attn = self.SelfAttention(q_attn)
        query_attn = self.CrossAttention(query_attn, q_attn, q_attn, OT=True).permute(0, 2, 1)
        query_attention = self.proj(query_attn)
        query_attention = query_attention.view(bsz, ch, h, w)

        support_res_feat, query_res_feat = self.heter_residual(support_attention, query_attention,s_local_res, q_local_res)
        # support_res_feat = support_attention
        # query_res_feat = query_attention
        with torch.no_grad():

            support_reshape = F.interpolate(support_res_feat, scale_factor=0.5, mode='bilinear')  # (4,1,25,25)
            s_multimodal = torch.mul(support_reshape, clip_refine_embed)
            s_multimodal = F.interpolate(s_multimodal, scale_factor=2, mode='bilinear')  # ([4, 1, 50, 50])

            query_reshape = F.interpolate(query_res_feat, scale_factor=0.5, mode='bilinear')  # (4,1,25,25)
            q_multimodal = torch.mul(query_reshape, clip_merge_embed)
            q_multimodal = F.interpolate(q_multimodal, scale_factor=2, mode='bilinear')  # ([4, 1, 50, 50])

        query_cam = query_cam.unsqueeze(1)
        support_cam = support_cam.unsqueeze(1)

        # decoder
        hypercorr_decoded_s = self.decoder1(s_multimodal+ after4d_support)

        upsample_size = (hypercorr_decoded_s.size(-1) * 2,) * 2
        hypercorr_decoded_s = F.interpolate(hypercorr_decoded_s, upsample_size, mode='bilinear', align_corners=True)
        logit_mask_support = self.decoder2(hypercorr_decoded_s)

        hypercorr_decoded_q = self.decoder1(q_multimodal + after4d_query)
        upsample_size = (hypercorr_decoded_q.size(-1) * 2,) * 2
        hypercorr_decoded_q = F.interpolate(hypercorr_decoded_q, upsample_size, mode='bilinear', align_corners=True)
        logit_mask_query = self.decoder2(hypercorr_decoded_q)

        logit_mask_support = self.res(
            torch.cat(
                [logit_mask_support, F.interpolate(support_cam, (100, 100), mode='bilinear', align_corners=True)],
                dim=1))
        logit_mask_query = self.res(
            torch.cat([logit_mask_query, F.interpolate(query_cam, (100, 100), mode='bilinear', align_corners=True)],
                      dim=1))

        # loss
        losses = 0
        if query_mask is not None:  # for training
            if not self.use_original_imgsize:
                logit_mask_query_temp = F.interpolate(logit_mask_query, support_img.size()[2:], mode='bilinear',
                                                      align_corners=True)
                logit_mask_support_temp = F.interpolate(logit_mask_support, support_img.size()[2:], mode='bilinear',
                                                        align_corners=True)

            loss_q = self.compute_objective(logit_mask_query_temp, query_mask) * 1.4
            loss_s = self.compute_objective(logit_mask_support_temp, support_mask) * 0.6

            lambda_l2 = 1e-4
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            l2_reg = lambda_l2 * l2_reg
            losses = loss_q + loss_s + l2_reg +losses

        if query_mask is not None:
            return logit_mask_query_temp, logit_mask_support_temp, losses
        else:
            # test
            if not self.use_original_imgsize:
                logit_mask_query = F.interpolate(
                    logit_mask_query, support_img.size()[2:], mode='bilinear', align_corners=True)
                logit_mask_support = F.interpolate(
                    logit_mask_support, support_img.size()[2:], mode='bilinear', align_corners=True)
            return logit_mask_query, logit_mask_support

    
    def reshape_features(self, low_feat, mid_feat, hig_feat):

        # 0-3 low layer：     torch.Size([bs, 512, 50, 50])
        # 4-9 middle layer：  torch.Size([bs, 1024, 25, 25])
        # 10-12 high layer：  torch.Size([bs, 2048, 13, 13])
        bsz,ch,h,w = mid_feat.size()
        # change ch
        if ch == 1024: # resnet
            low_feat = self.conv_512_128(low_feat)
            mid_feat = self.conv_1024_128(mid_feat)
            hig_feat = self.conv_2048_128(hig_feat)
        else: # vgg
            low_feat = self.conv_512_128(low_feat)
            mid_feat = self.conv_512_128(mid_feat)
            hig_feat = self.conv_512_128(hig_feat)

        # change size
        mid_feat = F.interpolate(mid_feat, (50, 50), mode='bilinear', align_corners=True)
        hig_feat = F.interpolate(hig_feat, (50, 50), mode='bilinear', align_corners=True)

        return low_feat, mid_feat, hig_feat

    def heter_residual(self, s_feat, q_feat, s_res, q_res):
        # Heterogeneous residuals
        s_res = F.interpolate(s_res, (100, 100), mode='bilinear', align_corners=True)
        q_res = F.interpolate(q_res, (100, 100), mode='bilinear', align_corners=True)
        _,ch,_,_ = s_res.size()
        if ch == 1024:  # resnet
            s_res = self.conv_1024_128(s_res)
            q_res = self.conv_1024_128(q_res)
        else: # vgg
            s_res = self.conv_512_128(s_res)
            q_res = self.conv_512_128(q_res)

        s_res = self.max_pool(s_res)

        q_res = self.avg_pool(q_res)

        s_feat = self.bn(s_feat)
        s_feat = s_feat + s_res
        s_feat = self.relu(s_feat)

        q_feat = self.bn(q_feat)
        q_feat = q_feat + q_res
        q_feat = self.relu(q_feat)

        return s_feat,q_feat

    def predict_mask_nshot(self, batch, nshot, class_id):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):

            logit_mask, logit_mask_s = self(query_img=batch['query_img'],
                                            support_img=batch['support_imgs'][:, s_idx],
                                            support_cam=batch['support_cams'],
                                            query_cam=batch['query_cam'],
                                            class_id=batch['class_id'])
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1:
                return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
