import cv2
import numpy as np
import open3d as o3d


def point_cloud_to_video(pcd_path, w=640, h=480):
    pcd = o3d.io.read_point_cloud(pcd_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    vis.add_geometry(pcd)
    ctrl = vis.get_view_control()
    ctrl.set_zoom(0.5)

    writer = cv2.VideoWriter(
        filename='results/indoor/video.mp4',
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=24,
        frameSize=(w, h),
    )

    n_frames = 400
    for i in range(n_frames):
        angle = 3.0 if i < int(0.25 * n_frames) or i > int(0.75 * n_frames) else -3.0
        ctrl.rotate(angle, 0.0)
        vis.poll_events()
        vis.update_renderer()
        frame = vis.capture_screen_float_buffer(False)
        frame = (255 * np.asarray(frame)).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    vis.destroy_window()


def compose_videos(input_video_path, pcd_video_path):
    cap1 = cv2.VideoCapture(input_video_path)
    cap2 = cv2.VideoCapture(pcd_video_path)

    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    dur1 = n1 / fps1
    dur2 = n2 / fps2
    duration = max(dur1, dur2)
    fps = 24
    n_frames = int(fps * duration)
    h, w = 720, 1280
    row_h = 600

    writer = cv2.VideoWriter(
        filename='results/demo.mp4',
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=fps,
        frameSize=(w, h),
    )

    def resize_to_height(frame, target_h):
        h, w = frame.shape[:2]
        scale = target_h / h
        return cv2.resize(frame, dsize=(int(w * scale), target_h))

    canvas = 255 * np.ones((h, w, 3), dtype=np.uint8)
    spacer = 255 * np.ones((row_h, 30, 3), dtype=np.uint8)
    for i in range(n_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        frame1 = resize_to_height(frame1, row_h)
        frame2 = resize_to_height(frame2, row_h)
        row = np.hstack((frame1, spacer, frame2))
        d = 60  # margin
        canvas[d: d + row.shape[0], d: d + row.shape[1]] = row
        writer.write(canvas)

    writer.release()
    cap1.release()
    cap2.release()


# point_cloud_to_video('results/indoor/cloud.ply')
# compose_videos('data/indoor.mov', 'results/indoor/video.mp4')
