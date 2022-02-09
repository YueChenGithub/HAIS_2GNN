import torch
import torch.nn as nn
import numpy as np

from models.basic_blocks import DynamicEdgeConv153

import torchsparse.nn as spnn

from models.basic_blocks import SparseConvEncoder_small
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
# from torchsparse.utils import sparse_collate_tensors
from torchsparse.utils.collate import sparse_collate


class RelationModule(nn.Module):
    def __init__(self, input_feature_dim, args, v_dim=128, h_dim=128, l_dim=256, dropout_rate=0.15):
        super().__init__()

        self.args = args
        self.input_feature_dim = input_feature_dim

        self.voxel_size = tuple(np.array([args.voxel_size_ap] * 3))
        # Sparse Volumetric Backbone
        self.net = SparseConvEncoder_small(self.input_feature_dim)
        self.pooling = spnn.GlobalMaxPool()

        self.vis_emb_fc = nn.Sequential(nn.Linear(v_dim, h_dim),
                                        nn.LayerNorm(h_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(h_dim, h_dim),
                                        )

        self.lang_emb_fc = nn.Sequential(nn.Linear(l_dim, h_dim),
                                         nn.BatchNorm1d(h_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(h_dim, h_dim),
                                         )
        # add input_feature_dim +3
        # self.gcn = DynamicEdgeConv(3 + input_feature_dim + args.num_classes, 128, k=args.k,
        #                            num_classes=args.num_classes)
        self.gcn = DynamicEdgeConv153(153, 128, k=args.k,
                                   num_classes=args.num_classes)
        self.one_hot_array = np.eye(args.num_classes)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def filter_candidates(self, data_dict, lang_feats, lang_cls_pred):
        instance_points = data_dict['instance_points']
        pred_obb_batch = data_dict['pred_obb_batch']
        instance_obbs = data_dict['instance_obbs']
        batch_size = len(instance_points)

        batch_index = []
        lang_feats_flatten = []
        pred_obbs = []
        feats = []
        filtered_index = []

        for i in range(batch_size):
            num_filtered_obj = len(pred_obb_batch[i])
            if num_filtered_obj < 2:
                continue

            lang_feat = lang_feats[i]  # (1, h_dim)
            lang_feat = lang_feat.repeat(num_filtered_obj, 1)
            lang_feats_flatten.append(lang_feat)

            instance_point = instance_points[i]
            instance_obb = instance_obbs[i]
            instance_class = data_dict['instance_class'][i]
            num_obj = len(instance_point)
            pred_obbs += list(instance_obb)

            # filter by class
            for j in range(num_obj):
                point_cloud = instance_point[j]
                point_cloud = point_cloud.mean(0)
                point_cloud[:3] = instance_obb[j][:3]  # change the mean average center to obb center (1,7)
                onhot_semantic = self.one_hot_array[instance_class[j]]
                point_cloud = np.concatenate([point_cloud, onhot_semantic], -1)  # (1,7+num_class)
                feats.append(point_cloud)
                if instance_class[j] == lang_cls_pred[i]:
                    filtered_index.append(len(batch_index))
                batch_index.append(i)

        '''
        B: Batch_size
        C: num_obj in a scene
        D: num_pred_obj
        feats: (C1*C2*...CB,7+num_class) 
               The feature of each obj in B scenes. One feat donates [x,y,z,a,b,c,d,onehot_semantic]
               xyz: center of obb
               abc: the mean of other features of 1024 sample points in the object
               onehot_semantic: [1,0,0,......,0] of len mum_class=18
        batch_index: (C1*C2*...*CB,)
                     to tell which object belongs to which scene (B=Batch_size=num_scene)
                     [0,0,0...,0,1,......31,31]
        filtered_index: 
        '''
        return feats, lang_feats_flatten, batch_index, filtered_index, pred_obbs

    def filter_candidates128(self, data_dict, lang_cls_pred):
        pred_obb_batch = []
        pts_batch = []
        obj_points_batch = []
        num_filtered_objs = []
        batch_size = len(data_dict['instance_points'])  # B

        pred_obb_batch1 = data_dict['pred_obb_batch'] #!!!
        for i in range(batch_size):

            #!!!
            num_filtered_obj = len(pred_obb_batch1[i])
            if num_filtered_obj < 2:
                continue

            instance_point = data_dict['instance_points'][i]  # (C,1024,7)
            instance_obb = data_dict['instance_obbs'][
                i]  # (C,7), [x,y,z,a,b,c,0], xyz: center, abc: size of bounding box
            instance_class = data_dict['instance_class'][i]  # (C,1)
            num_obj = len(instance_point)  # C

            pts = []
            pred_obbs = []

            # filter by class
            for j in range(num_obj):
                #if instance_class[j] == lang_cls_pred[i]:  # find the obj that match the language
                if 1:  # for all obj
                    pred_obbs.append(instance_obb[j])
                    point_cloud = instance_point[j]  # (1024,7)
                    pc = point_cloud[:, :3]  # (1024,3)
                    # fix sparse func

                    coords, indices = sparse_quantize(pc, voxel_size=self.voxel_size,
                                                      return_index=True)  # voxelize the points
                    feats = torch.tensor(point_cloud[:, 3:][indices], dtype=torch.float)
                    pt = SparseTensor(feats, coords)  # point tensor, feats: (D,4), coords: (D,3), D:num_voxel

                    pts.append(pt)
                    obj_points_batch.append(point_cloud)

            num_filtered_objs.append(len(pts))  # find the number of pred (obj that match the language) for each scene, len(pts): num_pred
            if len(pts) < 2:
                pts = []
            pts_batch += pts  # [pts(B=0), pts(B=1),...], pts(B=0) = [pt,pt,...]
            pred_obbs = np.asarray(pred_obbs)  # (num_pred,7)
            pred_obb_batch.append(pred_obbs)  # (num_pred,7)) for B batches

        """
        pts_batch: [pts(B=0), pts(B=1),...], pts(B=0) = [pt,pt,...] belongs to (num_pred, #pt)
        pred_obb_batch: [pred_obb(B=0), pred_obb(B=1),...], pred_obb: (num_pred,7)
        num_filtered_objs = [num_pred(B=0),...]
        """
        return pts_batch, pred_obb_batch, num_filtered_objs

    def get_feats128(self, data_dict):
        instance_points = data_dict['instance_points']  # (B,C,1024,7), B:batch_size, C: num_obj in a scene
        batch_size = len(instance_points)  # B



        # filter candidates
        # to get the classification of the to be predicted object according to language
        if not self.args.use_gt_lang:
            lang_scores = data_dict["lang_scores"]
            lang_cls_pred = torch.argmax(lang_scores, dim=1)
        else:
            lang_cls_pred = data_dict['object_cat']

        pts_batch, pred_obb_batch, num_filtered_objs = self.filter_candidates128(data_dict, lang_cls_pred)
        # pts_batch: list of the pts of all objects for each batch
        # pred_obb_batch: list of the obb of all objects for each batch
        # num_filtered_objs: list of the number of objects for each batch



        if pts_batch == []:
            return 0
        feats_128 = sparse_collate(pts_batch).cuda()  # get the feature of all objects

        # Sparse Volumetric Backbone
        feats_128 = self.net(feats_128)
        feats_128 = self.pooling(feats_128)  # (num_filtered_obj, 128), num_filtered_obj = sum(num_obj) in a batch


        return feats_128

    def forward(self, data_dict):
        lang_feats = data_dict['lang_rel_feats']  # (B, l_dim)
        lang_feats = self.lang_emb_fc(lang_feats).unsqueeze(1)  # (B, 1, h_dim)

        if not self.args.use_gt_lang:
            lang_scores = data_dict["lang_scores"]
            lang_cls_pred = torch.argmax(lang_scores, dim=1)
        else:
            lang_cls_pred = data_dict['object_cat']

        feats, lang_feats_flatten, batch_index, filtered_index, pred_obbs = \
            self.filter_candidates(data_dict, lang_feats, lang_cls_pred)

        lang_feats_flatten = torch.cat(lang_feats_flatten, dim=0)
        feats = torch.Tensor(feats).cuda()


        feats128 = self.get_feats128(data_dict) #!!!

        feats153 = torch.concat((feats128, feats), 1)

        batch_index = torch.LongTensor(batch_index).cuda()
        filtered_index = torch.LongTensor(filtered_index).cuda()
        support_xyz = torch.Tensor(pred_obbs)[:, :3].cuda()
        feats = self.gcn(support_xyz, batch_index, filtered_index, feats153)

        feats = self.vis_emb_fc(feats)

        scores = nn.functional.cosine_similarity(feats, lang_feats_flatten, dim=1)

        data_dict['relation_scores'] = scores

        return data_dict
