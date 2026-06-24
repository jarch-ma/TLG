r""" AFANet testing code  """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

from model.TLG import tlg
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

import clip

from thop import profile

    
class PredictMaskWrapper(nn.Module):
    def __init__(self, model, nshot):
        super().__init__()
        self.model = model
        self.nshot = nshot

    def forward(self, query_img, support_imgs, support_cam, query_cams, class_id):
        
            
        batch = {
            'query_img': query_img,
            'support_imgs': support_imgs,
            'support_cams': support_cam,
            'query_cam': query_cams,
            'class_id': class_id
        }
        pred_mask = self.model.predict_mask_nshot(batch, nshot=self.nshot, class_id=class_id)
        return pred_mask 







def test(model, dataloader, nshot):
    r""" Test TLG """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(6776)
    average_meter = AverageMeter(dataloader.dataset)

   # 计算flops 
    has_flop_logged = False
    for idx, batch in enumerate(dataloader):

        
        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)
        
        #######flops
        if not has_flop_logged:
            
            
            query_img = batch['query_img']
            support_imgs = batch['support_imgs']
            support_cams = batch['support_cams']
            query_cam = batch['query_cam']
            class_id = batch['class_id']

            input_data = (query_img, support_imgs, support_cams, query_cam, class_id)

            wrapper = PredictMaskWrapper(model.module, nshot)
            
            # 使用thop计算FLOPs
            thop_flops, thop_params = profile(
                wrapper,
                inputs=input_data,
                custom_ops={}
            )
            
            # 打印结果（thop的flops单位是FLOPs）
            print(f"总thop FLOPs: {thop_flops / 1e9:.2f} GFLOPs")
            print(f"总thop 参数: {thop_params / 1e6:.2f} M")

            has_flop_logged = True
            
        ############



        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot, class_id=batch['class_id'])  # AFANet

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if args.vis:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask,
                                                  batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float(),
                                                  batch['query_name'])

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Through the Looking Glass: A Dual Perspective on Few-Shot Semantic Segmentation')
    parser.add_argument('--datapath', type=str, default='/your_data_path/tlg_data/')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)  # must be 1
    parser.add_argument('--nworker', type=int, default=32)
    parser.add_argument('--load', type=str,
                        default='/your_data_path/tlg_data/model/voc_vgg_fold0/best_model.pt')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=5)
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'])
    parser.add_argument('--vis', default= False)
    parser.add_argument('--use_original_imgsize', action='store_true')  # action='store_true'

    # Pseudo mask for pascal dataset,
    parser.add_argument('--traincampath', type=str,
                        default='/your_data_path/tlg_data/voc_pesudo_mask/PseMask_Train/')
    parser.add_argument('--valcampath', type=str,
                        default='/your_data_path/tlg_data/voc_pesudo_mask/PseMask_Val/')

    # Pseudo mask for coco dataset
    # parser.add_argument('--traincampath', type=str, default='/your_data_path/coco_pesudo_mask/')
    # parser.add_argument('--valcampath', type=str, default='/your_data_path/coco_pesudo_mask/')'''

    parser.add_argument('--vispath', type=str, default='/your_data_path/vis/')

    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    clip_model, _ = clip.load('/your_data_path/tlg_pretrain/ViT-B-32.pt', device=device, jit=False)

    model = tlg(args.backbone, args.use_original_imgsize, args.benchmark, clip_model)  # TLG model
    model.eval()

    Logger.log_params(model)
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load, weights_only=True))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.vis, args.vispath)

    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.nshot,
                                                  cam_train_path=args.traincampath, cam_val_path=args.valcampath)
    # Test TLG

    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)

    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')