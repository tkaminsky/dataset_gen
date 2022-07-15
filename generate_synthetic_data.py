import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import argparse
import numpy as np
import random
import trimesh
import yaml
from collections import defaultdict
from munch import DefaultMunch
import nvisii as visii
import cv2
import h5py
import json
import copy

from visii_tools.loader.dict_loader import DictLoader
from visii_tools.scene import scene
from visii_tools.render import render_utils
import visii_tools
from visii_utils import sample_camera_position, set_assigned_materials, create_bin


TABLE_DIMS = [0.3 , 0.25, 0.1 ]
MAX_HEIGHT = 0.23
MAX_PADDING = 0.04


manager = trimesh.collision.CollisionManager()


def get_zoom_K(in_size, out_size, bbox, K, expand_ratio=1.0):
    a = get_zoom_factor(in_size, out_size, bbox, expand_ratio)
    K_new = np.zeros([3,3])
    K_new[:2, :] = a @ K
    K_new[2, 2] = 1
    return K_new


def get_zoom_factor(in_size, out_size, bbox, expand_ratio):
    in_height, in_width = in_size
    out_height, out_width = out_size
    ratio = out_height / out_width
    obj_start_x, obj_start_y, obj_end_x, obj_end_y = bbox
    zoom_c_x = 0.5*(obj_start_x+obj_end_x)
    zoom_c_y = 0.5*(obj_start_y+obj_end_y)

    left_dist = zoom_c_x - obj_start_x
    right_dist = obj_end_x - zoom_c_x
    up_dist = zoom_c_y - obj_start_y
    down_dist = obj_end_y - zoom_c_y

    crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * expand_ratio * 2
    crop_width = crop_height / ratio

    x1 = (zoom_c_x - crop_width / 2)
    x2 = (zoom_c_x + crop_width / 2)
    y1 = (zoom_c_y - crop_height / 2)
    y2 = (zoom_c_y + crop_height / 2)

    pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
    pts2 = np.float32([[0, 0], [0, out_size[0]], [out_size[1], 0]])
    affine_matrix = cv2.getAffineTransform(pts1, pts2)
    return affine_matrix


def project_point_to_image(p, camera, w, h):
    cam_matrix = camera.get_transform().get_world_to_local_matrix()
    cam_proj_matrix = camera.get_camera().get_projection()
    p_image = cam_proj_matrix * (cam_matrix * p)
    p_image = visii.vec2(p_image) / p_image.w
    p_image = p_image * visii.vec2(1,-1)
    p_image = (p_image + visii.vec2(1,1)) * 0.5
    return [p_image[0]*w, p_image[1]*h]


def setup_default_scene(scene_data, texture=None):
    mesh_root_path = ''
    loader = DictLoader(scene_data, mesh_root_path)
    entity_list = loader.entity_list  # list of entities to export
    create_bin(scene_data['scene']['table_dims'])
    scene.add_default_camera()
    set_assigned_materials(scene_data, texture)
    scene.add_default_lights()


def setup_camera(camera, pos_camera_random):

    entity = visii.entity.get('floor')
    obj_center = visii_tools.utils.vec_to_numpy(entity.get_aabb_center())
    camera.get_transform().look_at(
        at=visii_tools.utils.to_visii_vec(obj_center),
        up=visii.vec3(0, 0, 1),
        eye=visii_tools.utils.to_visii_vec(obj_center + pos_camera_random),
    )
        
    K = np.array([[921.5787, 0, 637.0372], [0, 919.6680, 362.2722], [0, 0, 1]])
    width=1280
    height=720
    top_left = project_point_to_image(visii.vec4(-0.185,0.15,0.275,1), camera, width, height)
    top_right = project_point_to_image(visii.vec4(-0.185,-0.15,0.275,1), camera, width, height)
    bot_right = project_point_to_image(visii.vec4(-0.185,-0.15,0.05,1), camera, width, height)
    bot_left = project_point_to_image(visii.vec4(-0.185,0.15,0.05,1), camera, width, height)

    x_min = min(top_left[0], bot_left[0])
    x_max = max(top_right[0], bot_right[0])

    y_min = min(top_left[1], top_right[1])
    y_max = max(bot_left[1], bot_right[1])

    K = get_zoom_K((1280,720), (256,256), (x_min, y_min, x_max, y_max), K)
    width=256
    height=256
    camera.get_camera().set_intrinsics(K[0,0], K[1,1], K[0,2], K[1,2], width, height)


def append_to_h5(h5_file, data0, mask0, depth0, data1, mask1, depth1, metadata0, metadata1):
    if 'frame0_data' not in h5_file:
        h5_file.create_dataset('frame0_data', data=data0, compression="gzip", chunks=True, maxshape=(None,256,256,4))
        h5_file.create_dataset('frame1_data', data=data1, compression="gzip", chunks=True, maxshape=(None,256,256,4))
        h5_file.create_dataset('frame0_mask', data=mask0, compression="gzip", chunks=True, maxshape=(None,256,256))
        h5_file.create_dataset('frame1_mask', data=mask1, compression="gzip", chunks=True, maxshape=(None,256,256))
        h5_file.create_dataset('frame0_depth', data=depth0, compression="gzip", chunks=True, maxshape=(None,256,256))
        h5_file.create_dataset('frame1_depth', data=depth1, compression="gzip", chunks=True, maxshape=(None,256,256))
        h5_file.create_dataset('frame0_metadata', data=metadata0, compression="gzip", chunks=True, maxshape=(None,))
        h5_file.create_dataset('frame1_metadata', data=metadata1, compression="gzip", chunks=True, maxshape=(None,))
    else:
        h5_file['frame0_data'].resize(h5_file['frame0_data'].shape[0] + 1, axis=0)
        h5_file['frame0_data'][-1] = data0
        h5_file['frame0_mask'].resize(h5_file['frame0_mask'].shape[0] + 1, axis=0)
        h5_file['frame0_mask'][-1] = mask0
        h5_file['frame0_depth'].resize(h5_file['frame0_depth'].shape[0] + 1, axis=0)
        h5_file['frame0_depth'][-1] = depth0
        h5_file['frame1_data'].resize(h5_file['frame1_data'].shape[0] + 1, axis=0)
        h5_file['frame1_data'][-1] = data1
        h5_file['frame1_mask'].resize(h5_file['frame1_mask'].shape[0] + 1, axis=0)
        h5_file['frame1_mask'][-1] = mask1
        h5_file['frame1_depth'].resize(h5_file['frame1_depth'].shape[0] + 1, axis=0)
        h5_file['frame1_depth'][-1] = depth1
        h5_file['frame0_metadata'].resize(h5_file['frame0_metadata'].shape[0] + 1, axis=0)
        h5_file['frame0_metadata'][-1:] = metadata0
        h5_file['frame1_metadata'].resize(h5_file['frame1_metadata'].shape[0] + 1, axis=0)
        h5_file['frame1_metadata'][-1] = metadata1


def get_metadata(scene_data, camera):
    json_scene_data = copy.deepcopy(scene_data)
    for k in json_scene_data['objects'].keys():
        json_scene_data['objects'][k]['transform']['transform'] = json_scene_data['objects'][k]['transform']['transform'].tolist()

    md = json.dumps({
        'name_to_id_map': dict(visii.texture.get_name_to_id_map()),
        'scene_data': json_scene_data,
        'camera': {
            'rotation': {
                'w': camera.get_transform().get_rotation().w,
                'x': camera.get_transform().get_rotation().x,
                'y': camera.get_transform().get_rotation().y,
                'z': camera.get_transform().get_rotation().z,
            },
            'position': {
                'x': camera.get_transform().get_position().x,
                'y': camera.get_transform().get_position().y,
                'z': camera.get_transform().get_position().z,
            },
            'intrinsic_matrix': visii_tools.utils.mat_to_numpy(camera.get_camera().get_intrinsic_matrix(256,256)).tolist()
        }
    })

    return md


def render(camera, width=256, height=256, spp=2000):
    img = render_utils.render_rgb(width, height, spp)
    seg = render_utils.render_segmentation(width, height)
    visii.set_camera_entity(camera)

    intrinsics = camera.get_camera().get_intrinsic_matrix(width, height)
    intrinsics = visii_tools.utils.mat_to_numpy(intrinsics)

    distance = render_utils.render_distance(width, height)
    depth = render_utils.depth_image_from_distance_image(
        distance, intrinsics)

    return img, seg, depth


def generate_dataset(args, folders, n, shard_prefix=''):
    # Group by the first two components of the name
    groups = defaultdict(list)
    for x in folders:
        groupname = '_'.join(x.split('_')[:2])
        groups[groupname].append(x)
    
    current_shard = 0
    current_shard_size = 0
    h5_file = None

    # List of possible box textures (ignore the commented-out ones)
    options = ['./cco_textures/PaintedPlaster004_2K-JPG/PaintedPlaster004_2K_Color.jpg', 
                './cco_textures/Sponge001_2K-JPG/Sponge001_2K_Color.jpg', 
                './cco_textures/Metal034_2K-JPG/Metal034_2K_Color.jpg',
                './cco_textures/Carpet011_2K-JPG/Carpet011_2K_Color.jpg',
                # './cco_textures/Gravel027_2K-JPG/Gravel027_2K_Color.jpg',
                './cco_textures/Marble026_2K-JPG/Marble026_2K_Color.jpg',
                # './cco_textures/TactilePaving001_2K-JPG/TactilePaving001_2K_Color.jpg',
                # './cco_textures/TactilePaving005_2K-JPG/TactilePaving005_2K_Color.jpg',
                './cco_textures/PaintedPlaster018_2K-JPG/PaintedPlaster018_2K_Color.jpg',
                './cco_textures/Paint006_2K-JPG/Paint006_2K_Color.jpg',
                './cco_textures/PaintedPlaster005_2K-JPG/PaintedPlaster005_2K_Color.jpg']

    # Determines whether the object is square enough to rotate more than 15 degrees
    THRESHOLD_ECCENTRICITY = 0.03

    for sample_i in range(n):
        # Let the user know how far along you are
        if sample_i % 10 == 0:
            print(f"\nOn {sample_i}/{n}\n")

        scene_data = {'scene': {}, 'objects': {}}
        scene_data['scene']['table_dims'] = TABLE_DIMS

        idx = np.random.randint(0, len(options))

        if h5_file is None:
            h5_file = h5py.File(os.path.join(args.output, f"{shard_prefix}_shard_{current_shard:06d}_prime.h5"), 'w')

        # Randomly shuffle list of groups
        group_keys = list(groups.keys())
        random.shuffle(group_keys)

        # Keep track of the current offset along the y axis
        y_offset = 0.15

        # Keep track of the objects we have added
        added_objects = []

        # Loop over 100 items. In practice, 100 items won't fit in the scene, so this loop should stop well before that.
        for i in range(100):

            # Get the group we are interested in adding to the scene. If args.same_group is true this will always be
            # the first group from the shuffled list. Otherwise we use a different group for each item.
            group_idx = 0 if args.same_group else i
            if group_idx >= len(group_keys):
                break
            groupname = group_keys[group_idx]

            # Randomly sample an item from the group and load its mesh
            item = random.choice(groups[groupname])
            obj_mesh = trimesh.load_mesh(f'{args.input}/{item}/meshes/model.obj')

            # Measure the mesh along each axis
            x_len = abs(obj_mesh.bounds[0,0] - obj_mesh.bounds[1,0])
            y_len = abs(obj_mesh.bounds[0,1] - obj_mesh.bounds[1,1])
            z_len = abs(obj_mesh.bounds[0,2] - obj_mesh.bounds[1,2])
            
            # Choose a random height within the allowable range
            size_max = MAX_HEIGHT
            size_min = 0.3 * MAX_HEIGHT
            scale = float(np.random.uniform(low=(size_min / z_len), high=(size_max / z_len)))
            x_len *= scale
            y_len *= scale
            z_len *= scale
            scale = [scale, scale, scale]

            # Initialize the transform matrix
            transform = np.identity(4)
            origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

            # Force book-like objects to be in the right place
            if round(y_len, 2) > round(x_len, 2):
                Rz = trimesh.transformations.rotation_matrix(1.57, zaxis)
                transform = trimesh.transformations.concatenate_matrices(transform, Rz)
                x_len_ = x_len
                x_len = y_len
                y_len = x_len_

            # If the object is approximately square, rotate it between -90 and 90 degrees
            if np.abs(round(y_len, 2) - round(x_len, 2)) < THRESHOLD_ECCENTRICITY:
                theta = np.random.uniform(low=-1.57, high=1.57)
                Rz2 = trimesh.transformations.rotation_matrix(theta, zaxis)
            # Otherwise, randomly rotate it between -15 and 15 degrees
            else:
                theta = np.random.uniform(low=-0.2618, high=0.2618)
                Rz2 = trimesh.transformations.rotation_matrix(theta, zaxis)\
            # Update the transform matrix and dimensions
            x_len_ = x_len
            y_len_ = y_len
            x_len = x_len_ * np.cos(np.abs(theta)) + y_len_ * np.sin(np.abs(theta))
            y_len = x_len_ * np.sin(np.abs(theta)) + y_len_ * np.cos(np.abs(theta))
            transform = trimesh.transformations.concatenate_matrices(transform, Rz2)

            # Randomly reflect the object along the y-z plane
            if np.random.uniform() > 0.5:
                Rz_reflect = trimesh.transformations.rotation_matrix(3.14, zaxis)
                transform = trimesh.transformations.concatenate_matrices(transform, Rz_reflect)
            
            # Filter objects that are too tall (this should never happen)
            if z_len >= MAX_HEIGHT:
                print("This should not happen")
                continue
            # If object is deeper than the box, scale it down
            if x_len >= 0.3:
                scale_factor = 0.3 / x_len
                scale = [scale[0] * scale_factor, scale[1] * scale_factor, scale[2] * scale_factor]
                x_len *= scale_factor
                y_len *= scale_factor
                z_len *= scale_factor
            # If object is too wide, scale it down
            if y_len >= 0.1:
                scale_factor = 0.1 / y_len
                scale = [scale[0] * scale_factor, scale[1] * scale_factor, scale[2] * scale_factor]
                x_len *= scale_factor
                y_len *= scale_factor
                z_len *= scale_factor

            translate = np.identity(4)
            translate[2,3] = 0.045
            transform = trimesh.transformations.concatenate_matrices(translate, transform)

            # Update our transform for this object to place it based on the current y_offset
            translate = np.identity(4)
            translate[1,3] = y_offset - (y_len/2)

            transform = trimesh.transformations.concatenate_matrices(translate, transform)
            if y_offset - y_len < -0.15:
                # This object doesn't fit, but one of the other ones might, so lets continue
                continue

            # Update our y_offset with the new object
            y_offset -= y_len

            # Add some random padding to the y offset
            if args.random_padding:
                y_offset -= random.uniform(0.0, MAX_PADDING)

            if y_offset < -0.15:
                # This means we have reached the end of the bin
                break

            scene_data['objects'][f'meshes_obj_{i}'] = {
                'entity': {
                    'type': 'MESH',
                    'file': f'{args.input}/{item}/meshes/model.obj',
                },
                'transform': {
                    'transform': transform,
                    'scale': scale
                },
                'metadata': {
                    'export': True,
                    'obj_path': f'{args.input}/{item}/meshes/model.obj',
                    'category': 'meshes',
                    'texture_path': f'{args.input}/{item}/materials/textures/texture.png'
                },
            }
            added_objects.append(i)
        
        if len(added_objects) == 0:
            print(f"Warning: empty scene")
            continue

        # Of all the added objects, we will randomly remove between 1-N of them.
        num_to_remove = random.randint(1, len(added_objects)) - 1

        # Randomly choose which objects will be removed
        objects_to_remove = random.sample(added_objects, num_to_remove)

        # Remove the objects from the scene
        for removal_i in objects_to_remove:
            del scene_data['objects'][f'meshes_obj_{removal_i}']
        
        # Setup render config
        render_cfg['camera_sampling']['radius']['min'] = 0.95
        render_cfg['camera_sampling']['radius']['max'] = 1.95
        render_cfg['camera']['spp'] = 2000
        render_cfg['camera']['width'] = 256
        render_cfg['camera']['height'] = 256

        # Render first frame
        setup_default_scene(scene_data, texture=options[idx])
        camera = visii.entity.get("camera")
        pos_camera_random = sample_camera_position(render_cfg, camera, look_at='floor')
        setup_camera(camera, pos_camera_random)

        img, seg, depth = render(camera)
        md = get_metadata(scene_data, camera)
        visii.texture_clear_all()
        visii.clear_all()


        # Add back the object to replace
        # scene_data['objects'][f'meshes_obj_new_{sample_i}'] = object_to_replace_data

        # # Render second frame
        # setup_default_scene(scene_data, texture=options[idx])
        # camera = visii.entity.get("camera")
        # setup_camera(camera, pos_camera_random)

        # img1, seg1, depth1 = render(camera)
        # md1 = get_metadata(scene_data, camera)
        # visii.texture_clear_all()
        # visii.clear_all()

        img1 = img
        seg1 = seg
        depth1 = depth
        md1 = md

        # Save to H5
        img = np.expand_dims(np.array(img), 0)
        img1 = np.expand_dims(np.array(img1), 0)
        seg = np.expand_dims(np.array(seg), 0)
        seg1 = np.expand_dims(np.array(seg1), 0)
        depth = np.expand_dims(np.array(depth), 0)
        depth1 = np.expand_dims(np.array(depth1), 0)
        
        append_to_h5(h5_file, img, seg, depth, img1, seg1, depth1, [md], [md1])

        current_shard_size += 1
        if current_shard_size >= args.shard_size:
            current_shard += 1
            h5_file = None
            current_shard_size = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input dataset directory', required=True)
    parser.add_argument('-o', '--output', help='Path to output directory', required=True)
    parser.add_argument('--num-holdout', help='Number of objects to hold out for test', default=1, type=int)
    parser.add_argument('--num-train-samples', help='Number of training samples to generate', default=1, type=int)
    parser.add_argument('--num-test-samples', help='Number of test samples to generate', default=1, type=int)
    parser.add_argument('--shard-size', help='Maximum number of samples per h5 shard', default=1, type=int)
    parser.add_argument('--same-group', action='store_true',
        help='When true, each generated scene contains all objects from the same group')
    parser.add_argument('--headless', action='store_true', help='Run visii headless')
    parser.add_argument('--random-padding', action='store_true', help='Add random padding between objects')
    args = parser.parse_args()

    random.seed(0)

    with open(os.path.dirname(os.path.realpath(__file__)) + '/cfg/google_scene.yaml', 'r') as file:
        render_cfg = DefaultMunch.fromDict(yaml.safe_load(file))

    # Read in list of files from the input path
    folders = os.listdir(args.input)

    random.shuffle(folders)

    switch = int(len(folders) * .174)

    test_folders = folders[:switch]
    train_folders = folders[switch:]
    print("Lengths")
    print(len(test_folders))
    print(len(train_folders))

    visii.initialize(headless=args.headless, verbose=True, lazy_updates=True)

    generate_dataset(args, train_folders, args.num_train_samples, 'train')
    generate_dataset(args, test_folders, args.num_test_samples, 'test')

    visii.deinitialize()
        
