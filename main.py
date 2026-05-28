import argparse
import cv2
import numpy as np
import open3d as o3d
import os
import torch
from tqdm import tqdm
from ultralytics import YOLO

from depth_anything_3.api import DepthAnything3


def load_video(path, max_frames=50, step=20, scale=0.2):
    cap = cv2.VideoCapture('data/' + path)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale)
            # cut the leftmost pixels, iphone videos have them black for some reason
            frame = frame[:, 5:, :]
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def estimate_depth(frames):
    """ Monocular depth estimation """
    model = DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE')
    model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    return model.inference(frames)


def segment_objects(model, frame):
    """ Instance segmentation """
    output = model.predict(frame, verbose=False)[0]
    color_map = frame.copy()
    if output.masks is not None:
        masks = output.masks.data.cpu().numpy().astype(bool)
        class_ids = output.boxes.cls.cpu().numpy().astype(int)
        labels = [output.names[c] for c in class_ids]
        colors = {}
        rng = np.random.default_rng(42)

        def get_color(label):
            if label not in colors:
                colors[label] = rng.integers(low=50, high=255, size=3, dtype=np.uint8)
            return colors[label]

        color_map = frame.copy()
        for mask, label in zip(masks, labels):
            if mask.shape != frame.shape[:2]:
                h, w = frame.shape[:2]
                mask = cv2.resize(
                    mask.astype(np.uint8), dsize=(w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            color = get_color(label)
            color_map[mask] = (0.7 * color + 0.3 * frame[mask]).astype(np.uint8)
    return color_map


def depth_to_point_cloud(frame, depth, intrinsic, extrinsic):
    """ Perspective projection """
    frame = np.ascontiguousarray(frame).astype(np.uint8)
    depth = np.ascontiguousarray(depth).astype(np.float32)
    color_o3d = o3d.geometry.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth)

    h, w = depth.shape
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    K =  o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    H = np.vstack([extrinsic, np.array([[0, 0, 0, 1]])])

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic=K, extrinsic=H
    )
    # flip upside-down
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    pcd.transform(flip)
    return pcd


# def point_cloud_to_mesh(pcd, voxel_size=0.05):
#     """ Poisson surface reconstruction """
#     # pcd = pcd.voxel_down_sample(voxel_size)
#     pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
#     pcd.orient_normals_consistent_tangent_plane(k=15)
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd, depth=9, linear_fit=False
#     )
#     # Transfer colors from the point cloud to the mesh
#     tree = o3d.geometry.KDTreeFlann(pcd)
#     vertex_colors = []
#     for v in np.asarray(mesh.vertices):
#         idx = tree.search_knn_vector_3d(v, 1)[1]
#         vertex_colors.append(np.asarray(pcd.colors)[idx[0]])
#
#     mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
#     # mesh.remove_duplicated_vertices()
#     # mesh.remove_duplicated_triangles()
#     # mesh.remove_degenerate_triangles()
#     return mesh


def save_point_cloud(pcd, input_path):
    output_path = f'results/{os.path.splitext(input_path)[0]}/'
    os.makedirs(output_path, exist_ok=True)
    success = o3d.io.write_point_cloud(output_path + 'cloud.ply', pcd, write_ascii=False)
    if not success:
        raise RuntimeError('Failed to save point cloud')


def main(args):
    print('Loading video...')
    frames = load_video(args.video_path)

    print('Estimating depth...')
    with torch.no_grad():
        pred = estimate_depth(frames)
    images = pred.processed_images

    color_maps = images.copy()
    if args.segment:
        model = YOLO('weights/yolo26n-seg.pt')
        color_maps = []
        for i in tqdm(range(len(frames)), desc='Segmenting instances...'):
            color_maps.append(segment_objects(model, images[i]))

    combined_pcd = o3d.geometry.PointCloud()
    for i in tqdm(range(len(frames)), desc='Converting to point cloud...'):
        pcd = depth_to_point_cloud(
            color_maps[i], pred.depth[i], pred.intrinsics[i], pred.extrinsics[i]
        )
        combined_pcd += pcd

    if args.save:
        save_point_cloud(combined_pcd, args.video_path)

    o3d.visualization.draw_geometries([combined_pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='indoor.mov',
        help='Path to the input video (if path is relative, file is taken from the "data" folder)'
    )
    parser.add_argument('--segment', action='store_true',
        help='Run instance segmentation'
    )
    parser.add_argument('--save', action='store_true',
        help='Save the final mesh to the "results" folder'
    )
    args = parser.parse_args()
    main(args)
