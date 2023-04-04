import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field


@dataclass
class Project3D(nn.module):
    """
    Project 2D depth map to 3D point cloud
    """
    batch_size: int
    height: int
    width: int
    pixel_coordinates: torch.Tensor = field(init=False)
    ones: torch.Tensor = field(init=False)

    def __post_init__(self):
        meshgrid: npt.ArrayLike = np.stack(np.meshgrid(range(self.width), range(self.height), indexing="xy"), axis=0)
        id_coordinates = nn.Parameter(torch.from_numpy(meshgrid.astype(np.float32)), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)
        self.pixel_coordinates = torch.unsqueeze(torch.stack([id_coordinates[0].view(-1), id_coordinates[1].view(-1)], 0), 0)
        self.pixel_coordinates = self.pixel_coordinates.repeat(self.batch_size, 1, 1)
        self.pixel_coordinates = nn.Parameter(torch.cat([self.pixel_coordinates, self.ones], 1), requires_grad=False)

    def forward(self, depth_map: torch.Tensor, inverse_K: torch.Tensor) -> torch.Tensor:
        """
        Project 2D depth map to 3D point cloud
        :param depth_map: (B, 1, H, W)
        :param inverse_K: (B, 3, 3)
        :return: (B, 3, H*W)
        """
        depth_map = depth_map.view(self.batch_size, 1, -1)
        point_cloud = torch.matmul(inverse_K, self.pixel_coordinates)
        point_cloud = point_cloud * depth_map
        point_cloud = torch.cat([point_cloud, self.ones], 1)
        return point_cloud
