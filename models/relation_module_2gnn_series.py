import torch
import torch.nn as nn
import numpy as np

from models.basic_blocks import DynamicEdgeConv, DynamicEdgeConv2


class RelationModule(nn.Module):
    def __init__(self, input_feature_dim, args, v_dim=256, h_dim=256, l_dim=256, dropout_rate=0.15):
        super().__init__()

        self.args = args
        self.input_feature_dim = input_feature_dim
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
        self.gcn = DynamicEdgeConv(3 + input_feature_dim + args.num_classes, 128, k=args.k,
                                   num_classes=args.num_classes)

        self.gcn2 = DynamicEdgeConv2(153, 128, k=args.k,
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
        filtered_index: [a,b,c...]
                       the index for objects that match the language
        pred_obbs: [obb(B0), obb(B1),...]
                  list of all objects obb
        '''
        return feats, lang_feats_flatten, batch_index, filtered_index, pred_obbs

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

        batch_index = torch.LongTensor(batch_index).cuda()
        filtered_index = torch.LongTensor(filtered_index).cuda()
        support_xyz = torch.Tensor(pred_obbs)[:, :3].cuda()
        feats_gnn1_output = self.gcn(support_xyz, batch_index, filtered_index, feats)

        feats_25 = torch.index_select(feats, 0, filtered_index)  # filtered feats (only for candidates)

        feat_gnn2_input = torch.concat((feats_gnn1_output, feats_25), dim=1)  #153

        feats_gnn2_output = self.gcn2(support_xyz, batch_index, filtered_index, feat_gnn2_input, batch_size=len(lang_feats), filt_features=False) #128

        feats = torch.concat((feats_gnn1_output, feats_gnn2_output), dim=1)

        feats = self.vis_emb_fc(feats)

        scores = nn.functional.cosine_similarity(feats, lang_feats_flatten, dim=1)

        data_dict['relation_scores'] = scores

        return data_dict
