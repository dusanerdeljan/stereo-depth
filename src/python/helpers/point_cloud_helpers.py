import open3d as o3d
import torch


def depth_to_o3d_point_cloud(depth: torch.Tensor, mask: torch.Tensor) -> o3d.geometry.PointCloud:
    depth_points = []
    for x in range(depth.shape[0]):
        for y in range(depth.shape[1]):
            if mask[x, y]:
                depth_points.append([y, x, depth[x, y]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth_points)
    return pcd


def visualize_point_cloud_from_depth(depth: torch.Tensor, mask: torch.Tensor) -> None:
    pcd = depth_to_o3d_point_cloud(depth, mask)
    o3d.visualization.draw_geometries([pcd])


def save_point_cloud_from_depth(depth: torch.Tensor, mask: torch.Tensor, filename: str) -> None:
    pcd = depth_to_o3d_point_cloud(depth, mask)
    o3d.io.write_point_cloud(filename, pcd)
