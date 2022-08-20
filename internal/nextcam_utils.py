"""Utilities for NextCamBundle datasets"""

import numpy as np
import skimage

from internal import image as lib_image


def extract_data(npz_file, scale_factor=1.):
  """Loads images, poses, intrinsics, depth, and confidences from disk."""

  bundle = np.load(npz_file, allow_pickle=True)
  num_frames = bundle["num_frames"]

  # Extract poses, intrinsics, images.
  images = []
  camtoworlds = []
  pixtocams = []
  depths = []
  confidences = []
  for i in range(num_frames):
    # Poses.
    w2c = bundle[f"info_{i}"].item()["world_to_camera"]
    camtoworlds.append(np.linalg.inv(w2c))

    # Intrinsics.
    # Swap x and y of principal point values in K.
    K_local = bundle[f"info_{i}"].item()["intrinsics"]
    camera_mat = K_local.copy()
    camera_mat[0, 2] = K_local[1, 2]
    camera_mat[1, 2] = K_local[0, 2]

    if scale_factor > 1:
      # Scale camera matrix according to downsampling factor.
      camera_mat = np.diag([1. / scale_factor, 1. / scale_factor, 1.]).astype(
          np.float32) @ camera_mat

    pixtocams.append(np.linalg.inv(camera_mat))

    # Images.
    image = (bundle[f"img_{i}"] / 255.).astype(np.float32)
    if scale_factor > 1:
      image = lib_image.downsample(image, scale_factor)
    images.append(image)

    depth = bundle[f"depth_{i}"]
    depth = skimage.transform.resize(depth, image.shape[:2])
    depths.append(depth)
    conf = bundle[f"conf_{i}"]
    conf = skimage.transform.resize(
        conf,
        image.shape[:2],
    )
    confidences.append(conf)

  images = np.stack(images)
  camtoworlds = np.stack(camtoworlds)
  pixtocams = np.stack(pixtocams)
  depths = np.stack(depths)
  confidences = np.stack(confidences)

  return images, camtoworlds, pixtocams, depths, confidences


def world_from_pix_and_z(pixtocam, camtoworld, depth, conf):
  """Calculate points in world coordinates for this camera."""
  # Make homogeneous.
  pixtocam_h = np.eye(4)
  pixtocam_h[:3, :3] = pixtocam
  pixel_grid = np.array(
      np.meshgrid(np.arange(depth.shape[1]),
                  np.arange(depth.shape[0]),
                  indexing='xy'))
  # Homogeneous coords: (u, v, 1, 1/z)
  # Shape: [4, H, W]
  pix_h = np.stack(
      [pixel_grid[0], pixel_grid[1],
       np.ones_like(depth), 1 / depth])
  # Shape [4, H*W]
  pix_h = np.reshape(pix_h, (pix_h.shape[0], np.prod(pix_h.shape[1:])))
  depth = np.reshape(depth, (np.prod(depth.shape)))
  conf = np.reshape(conf, (np.prod(conf.shape)))

  # Convert c2w from OpenGL to OpenCV
  camtoworld = camtoworld @ np.diag([1, -1, -1, 1])

  # Unproject.
  points = camtoworld @ pixtocam_h @ pix_h * depth

  # Only return high confidence points.
  return points[:, conf * 255 > 1.5]


def get_all_world_points(pixtocams, camtoworlds, depths, confs):
  """Get all reliable 3D points in world space by unprojecting depth maps."""
  all_selected_points = []
  for i in range(pixtocams.shape[0]):
    selected_points = world_from_pix_and_z(pixtocams[i], camtoworlds[i],
                                           depths[i], confs[i])
    all_selected_points.append(selected_points)

  return np.concatenate(all_selected_points, axis=1)