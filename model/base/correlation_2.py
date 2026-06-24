r""" Provides functions that builds/manipulates correlation tensors """
import torch


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):   # support 是二值化mask
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1) # (bsz,ch, h*w)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps) # 归一化 

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            # 每一层矩阵都相乘 生成4维关联矩阵 C
            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0) # 截断，最小为0
            corrs.append(corr)

        # l4 高层: 10,11,12    l3 中层: 4,5,6,7,8,9    l2 低层: 0,1,2,3
        # 8，3，13，13，13，13
        # corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        # 8，6，25，25，25，25
        # corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        # 8，4，50，50，50，50
        # corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        # 50,8,50,50,50 -> 8,1,13,13,13,13
        corr_l4 = corrs[stack_ids[2]].contiguous().unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)
        # 25，8，25，25，25 -> 8,1,25,25,25,25
        corr_l3 = corrs[stack_ids[1]].contiguous().unsqueeze(1).repeat(1, 6, 1, 1, 1, 1)
        # 13，8，13，13，13 -> 8,1,13,13,13,13
        corr_l2 = corrs[stack_ids[0]].contiguous().unsqueeze(1).repeat(1, 4, 1, 1, 1, 1)


        return [corr_l4, corr_l3, corr_l2]   # 只获取了高层特征
