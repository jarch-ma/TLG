r""" PASCAL-5i few-shot semantic segmentation dataset """
import os
import pdb

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, cam_train_path, cam_val_path):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'voc2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'voc2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.cam_train_path = cam_train_path
        self.cam_val_path = cam_val_path

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_gt_mask, support_imgs, support_gt_masks, org_qry_imsize = \
            self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_gt_mask = F.interpolate(query_gt_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:],
                                        mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_gt_mask.float(), class_sample)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for support_gt_mask in support_gt_masks:
            support_gt_mask = F.interpolate(support_gt_mask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                   mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(support_gt_mask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        if self.split == 'val': # pse mask

            query_pse_mask_path = self.cam_val_path + query_name + '--' + str(class_sample) + '.png'

            query_pse_mask = torch.from_numpy(np.array(Image.open(query_pse_mask_path))).float()
            query_pse_mask = self.extract_pse_mask_label(query_pse_mask, class_sample)
            query_pse_mask = F.interpolate(query_pse_mask.unsqueeze(0).unsqueeze(0), (400, 400),mode='bilinear')
            query_pse_mask = self.filter_pse_mask(query_pse_mask)

            nshot = len(support_names)

            support_pse_masks = []
            for nn in range(nshot):

                support_pse_mask_path = self.cam_val_path + support_names[nn] + '--' + str(class_sample) + '.png'

                support_pse_mask = torch.from_numpy(np.array(Image.open(support_pse_mask_path))).float()
                support_pse_mask = self.extract_pse_mask_label(support_pse_mask, class_sample)
                support_pse_mask = F.interpolate(support_pse_mask.unsqueeze(0).unsqueeze(0), (400, 400), mode='bilinear')
                support_pse_mask = self.filter_pse_mask(support_pse_mask)

                support_pse_masks.append(support_pse_mask)

            support_pse_masks = torch.cat(support_pse_masks, dim=0)
        else:  # pse train mask

            query_pse_mask_path = self.cam_train_path + query_name + '--' + str(class_sample) + '.png'

            query_pse_mask = torch.from_numpy(np.array(Image.open(query_pse_mask_path))).float()
            query_pse_mask = self.extract_pse_mask_label(query_pse_mask, class_sample)
            query_pse_mask = F.interpolate(query_pse_mask.unsqueeze(0).unsqueeze(0), (400, 400), mode='bilinear')
            query_pse_mask = self.filter_pse_mask(query_pse_mask)

            nshot = len(support_names)

            support_pse_masks = []
            for nn in range(nshot):

                support_pse_mask_path = self.cam_train_path + support_names[nn] + '--' + str(class_sample) + '.png'

                support_pse_mask = torch.from_numpy(np.array(Image.open(support_pse_mask_path))).float()
                support_pse_mask = self.extract_pse_mask_label(support_pse_mask, class_sample)
                support_pse_mask = F.interpolate(support_pse_mask.unsqueeze(0).unsqueeze(0), (400, 400), mode='bilinear')
                support_pse_mask = self.filter_pse_mask(support_pse_mask)
                support_pse_masks.append(support_pse_mask)


            support_pse_masks = torch.cat(support_pse_masks, dim=0)
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,
                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,
                 'class_id': torch.tensor(class_sample),
                 'query_cam': query_pse_mask,
                 'support_cams': support_pse_masks}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def extract_pse_mask_label(self, pse_mask, class_id):
        # boundary = (mask / 255).floor()
        pse_mask[pse_mask != class_id + 1] = 0
        pse_mask[pse_mask == class_id + 1] = 1

        return pse_mask

    def filter_pse_mask(self, pse_mask):
        # boundary = (mask / 255).floor()
        pse_mask[pse_mask >=0.5] = 1
        pse_mask[pse_mask <0.5] = 0

        return pse_mask.squeeze().squeeze()

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('/media/newssd2/jiaqi/code/TLG_final_thu/data/splits/pascal//%s//fold%d.txt' % (split, fold_id)) # 现在更推荐f-string或者 format写法

            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]

            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)

        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)

        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]

        return img_metadata_classwise
