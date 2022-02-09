import math
import torch
import torch.nn as nn
import torchsparse.nn as spnn

from torchsparse import SparseTensor
from torch_geometric.nn import MessagePassing, knn

# replace transpose with transposed
class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, transposed=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,
                        transposed=transposed),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SparseConvEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, ks=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

class SparseConvEncoder_small(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, ks=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )




    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)


        return x




class DynamicEdgeConv(MessagePassing):
    def __init__(self, F_in, F_out, k=6, num_classes=18):
        super(DynamicEdgeConv, self).__init__(aggr='max')
        self.k = k
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(3 * F_in, F_out),
            nn.ReLU(),
            nn.Linear(F_out, F_out)
        )
        self.weight = nn.Sequential(
            nn.Linear(3+num_classes+num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, F_in)
        )

    def forward(self, support_xyz, batch_index, filtered_index, features):
        # knn
        query_xyz = torch.index_select(support_xyz, 0, filtered_index)  # (C1*C2*...*CB,3)
        query_batch_index = torch.index_select(batch_index, 0, filtered_index)  # (C1*C2*...*CB,), [0,0,0,...0,1,...,1,2,...,15]
        query_features = torch.index_select(features, 0, filtered_index)

        row, col = knn(support_xyz, query_xyz, self.k, batch_index, query_batch_index)
        edge_index = torch.stack([col, row], dim=0)

        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=(features, query_features), pos=(support_xyz, query_xyz))  # shape [N, F_out]

    def message(self, x_i, x_j, pos_i, pos_j):
        # print(pos_j)
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_weights = self.weight(torch.cat([pos_j - pos_i, x_i[:,-self.num_classes:], x_j[:,-self.num_classes:]],-1))
        # print("shapes", x_i.shape,edge_weights.shape,x_j.shape)
        edge_features = torch.cat([x_i, edge_weights, x_j], dim=1)  # shape [E, 3 * F_in]
        
        return self.mlp(edge_features)  # shape [E, F_out]


class DynamicEdgeConv153(MessagePassing):
    def __init__(self, F_in, F_out, k=6, num_classes=18):
        super(DynamicEdgeConv153, self).__init__(aggr='max')
        self.k = k
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(3 * F_in, F_out),
            nn.ReLU(),
            nn.Linear(F_out, F_out)
        )
        self.weight = nn.Sequential(
            nn.Linear(3 + num_classes + num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, F_in)
        )

    def forward(self, support_xyz, batch_index, filtered_index, features):
        # knn
        query_xyz = torch.index_select(support_xyz, 0, filtered_index)  # find the xyz of obj that match the language
        query_batch_index = torch.index_select(batch_index, 0, filtered_index)
        query_features = torch.index_select(features, 0, filtered_index)

        row, col = knn(support_xyz, query_xyz, self.k, batch_index, query_batch_index)  # find k neighbor in support_xyz for each of query_xyz
        edge_index = torch.stack([col, row], dim=0)



        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=(features, query_features),
                              pos=(support_xyz, query_xyz))  # shape [N, F_out]

    def message(self, x_i, x_j, pos_i, pos_j):
        # print(pos_j)
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_weights = self.weight(
            torch.cat([pos_j - pos_i, x_i[:, -self.num_classes:], x_j[:, -self.num_classes:]], -1))  # x_i[:, -self.num_classes:] represent the class of each object
        # print("shapes", x_i.shape,edge_weights.shape,x_j.shape)
        edge_features = torch.cat([x_i, edge_weights, x_j], dim=1)  # shape [E, 3 * F_in]

        return self.mlp(edge_features)  # shape [E, F_out]

class DynamicEdgeConv2(MessagePassing):
    def __init__(self, F_in, F_out, k=6, num_classes=18):
        super(DynamicEdgeConv2, self).__init__(aggr='max')
        self.k = k
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(3 * F_in, F_out),
            nn.ReLU(),
            nn.Linear(F_out, F_out)
        )
        self.weight = nn.Sequential(
            nn.Linear(3 + num_classes + num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, F_in)
        )

    def forward(self, support_xyz, batch_index, filtered_index, features, batch_size=32, filt_features=True):

        query_xyz = torch.index_select(support_xyz, 0, filtered_index)  # find the xyz of obj that match the language # (C1*C2*...*CB,3)
        query_batch_index = torch.index_select(batch_index, 0, filtered_index)  # find the batch_index of obj that match the language # (C1*C2*...*CB,), [0,0,0,...0,1,...,1,2,...,15]
        if filt_features:
            query_features = torch.index_select(features, 0, filtered_index)  # find the features of obj that match the language
        else:
            query_features = features
        num_query = query_batch_index.bincount(minlength=batch_size)  # get the num_query for each batch. (C1,C2,....)

        num_query_sum = torch.zeros(batch_size+1, dtype=int)
        for i in range(batch_size + 1):
            num_query_sum[i] = num_query[:i].sum()  # [0,C1,C1+C2,C1+C2+C3,...]

        query_index = []
        for i in range(len(query_batch_index)):
            query_index.append(i)  # (C1*C2*...*CB,), [0,1,2,......]
        query_index = torch.tensor(query_index)

        row = []
        col = []
        for i in range(batch_size):
            query_index_for_batch = query_index[num_query_sum[i]:num_query_sum[i + 1]]
            a, b = torch.meshgrid(query_index_for_batch, query_index_for_batch)
            row.append(a.flatten())
            col.append(b.flatten())

        row = torch.cat(row)
        col = torch.cat(col)

        #remove self loop
        mask_self_loop = torch.ones(len(row), dtype=int)
        for i in range(len(row)):
            if row[i] == col[i]:
                mask_self_loop[i] = 0
        row = row[mask_self_loop.bool()]
        col = col[mask_self_loop.bool()]

        row = row.cuda()
        col = col.cuda()

        edge_index = torch.stack([col, row], dim=0)

        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=(query_features, query_features),
                              pos=(query_xyz, query_xyz))  # shape [N, F_out]

    def message(self, x_i, x_j, pos_i, pos_j):
        # print(pos_j)
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_weights = self.weight(
            torch.cat([pos_j - pos_i, x_i[:, -self.num_classes:], x_j[:, -self.num_classes:]], -1))  # x_i[:, -self.num_classes:] represent the class of each object
        # print("shapes", x_i.shape,edge_weights.shape,x_j.shape)
        edge_features = torch.cat([x_i, edge_weights, x_j], dim=1)  # shape [E, 3 * F_in]

        return self.mlp(edge_features)  # shape [E, F_out]

class BEVEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, ks=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.stage4(x)
        return x


def spcrop(inputs, loc_min, loc_max):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    valid_flag = ((coords[:, :3] >= loc_min) & (coords[:, :3] < loc_max)).all(-1)
    output_coords = coords[valid_flag]
    output_features = features[valid_flag]
    return SparseTensor(output_features, output_coords, cur_stride)


class SparseCrop(nn.Module):
    def __init__(self, loc_min, loc_max):
        super().__init__()
        self.loc_min = loc_min
        self.loc_max = loc_max

    def forward(self, inputs):
        return spcrop(inputs, self.loc_min, self.loc_max)


class ToDenseBEVConvolution(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape,
                 offset: list = [0, 0, 0],
                 z_dim: int = 1,
                 use_bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset = torch.cuda.IntTensor([list(offset) + [0]])
        self.z_dim = z_dim
        self.n_kernels = int(shape[self.z_dim])
        self.bev_dims = [i for i in range(3) if i != self.z_dim]
        self.bev_shape = shape[self.bev_dims]
        self.kernel = nn.Parameter(torch.zeros(self.n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if use_bias else 0
        self.init_weight()

    def __repr__(self):
        return 'ToDenseBEVConvolution(in_channels=%d, out_channels=%d, n_kernels=%d)' % (
            self.in_channels,
            self.out_channels,
            self.n_kernels
        )

    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        # print(cur_stride, self.z_dim,self.n_kernels,self.kernel.shape)
        kernels = torch.index_select(self.kernel, 0, coords[:, self.z_dim].long() // cur_stride[0])
        sparse_features = (features.unsqueeze(-1) * kernels).sum(1) + self.bias
        sparse_coords = (coords - self.offset).t()[[3] + self.bev_dims].long()
        sparse_coords[1:] = sparse_coords[1:] // cur_stride[0]
        batch_size = sparse_coords[0].max().item() + 1
        sparse_coords = sparse_coords[0] * int(self.bev_shape.prod()) + sparse_coords[1] * int(self.bev_shape[1]) + \
                        sparse_coords[2]
        bev = torch.cuda.sparse.FloatTensor(
            sparse_coords.unsqueeze(0),
            sparse_features,
            torch.Size([batch_size * int(self.bev_shape.prod()), sparse_features.size(-1)]),
        ).to_dense()
        return bev.view(batch_size, *self.bev_shape, -1).permute(0, 3, 1, 2).contiguous()  # To BCHW


def tensor2points(tensor, offset=(-80., -80., -5.), voxel_size=(.05, .05, .1)):
    indices = tensor.float()
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, :3] = indices[:, :3] * voxel_size + offset + .5 * voxel_size
    return indices

# from lib.pointnet2.pointnet2_modules import PointnetSAModule

# def break_up_pc(pc: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
#     """
#     Split the pointcloud into xyz positions and features tensors.
#     This method is taken from VoteNet codebase (https://github.com/facebookresearch/votenet)
#
#     @param pc: pointcloud [N, 3 + C]
#     :return: the xyz tensor and the feature tensor
#     """
#     xyz = pc[..., 0:3].contiguous()
#     features = (
#         pc[..., 3:].transpose(1, 2).contiguous()
#         if pc.size(-1) > 3 else None
#     )
#     return xyz, features
#
#
# class PointNetPP(nn.Module):
#     """
#     Pointnet++ encoder.
#     For the hyper parameters please advise the paper (https://arxiv.org/abs/1706.02413)
#     """
#
#     def __init__(self, sa_n_points: list,
#                  sa_n_samples: list,
#                  sa_radii: list,
#                  sa_mlps: list,
#                  bn=True,
#                  use_xyz=True):
#         super().__init__()
#
#         n_sa = len(sa_n_points)
#         if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
#             raise ValueError('Lens of given hyper-params are not compatible')
#
#         self.encoder = nn.ModuleList()
#
#         for i in range(n_sa):
#             self.encoder.append(PointnetSAModule(
#                 npoint=sa_n_points[i],
#                 nsample=sa_n_samples[i],
#                 radius=sa_radii[i],
#                 mlp=sa_mlps[i],
#                 bn=bn,
#                 use_xyz=use_xyz,
#             ))
#
#         out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
#         self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1], sa_mlps[-1][-1])
#
#     def forward(self, features):
#         """
#         @param features: B x N_objects x N_Points x 3 + C
#         """
#         xyz, features = break_up_pc(features)
#         for i in range(len(self.encoder)):
#             xyz, features = self.encoder[i](xyz, features)
#
#         return self.fc(features.view(features.size(0), -1))
