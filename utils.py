import torch
import numpy as np
import cv2




def convert2np(x):
    # import ipdb; ipdb.set_trace()
    # if type(x) is torch.autograd.Variable:
    #     x = x.data
    # Assumes x is gpu tensor..
    if type(x) is not np.ndarray:
        return x.cpu().numpy()
    return x


def tensor2mask(image_tensor, imtype=np.uint8):
    # Input is H x W
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.expand_dims(image_numpy, 2) * 255.0
    image_numpy = np.tile(image_numpy, (1, 1, 3))
    return image_numpy.astype(imtype)


def kp2im(kp, img, radius=None):
    """
    Input is numpy array or torch.cuda.Tensor
    img can be H x W, H x W x C, or C x H x W
    kp is |KP| x 2

    """
    kp_norm = convert2np(kp)
    img = convert2np(img)

    if img.ndim == 2:
        img = np.dstack((img, ) * 3)
    # Make it H x W x C:
    elif img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:  # Gray2RGB for H x W x 1
            img = np.dstack((img, ) * 3)

    # kp_norm is still in [-1, 1], converts it to image coord.
    kp = (kp_norm[:, :2] + 1) * 0.5 * img.shape[0]
    if kp_norm.shape[1] == 3:
        vis = kp_norm[:, 2] > 0
        kp[~vis] = 0
        kp = np.hstack((kp, vis.reshape(-1, 1)))
    else:
        vis = np.ones((kp.shape[0], 1))
        kp = np.hstack((kp, vis))

    kp_img = draw_kp(kp, img, radius=radius)

    return kp_img


def draw_kp(kp, img, radius=None):
    """
    kp is 15 x 2 or 3 numpy.
    img can be either RGB or Gray
    Draws bird points.
    """
    if radius is None:
        radius = max(4, (np.mean(img.shape[:2]) * 0.01).astype(int))

    num_kp = kp.shape[0]
    # Generate colors
    import pylab
    cm = pylab.get_cmap('gist_rainbow')
    colors = 255 * np.array([cm(1. * i / num_kp)[:3] for i in range(num_kp)])
    white = np.ones(3) * 255

    image = img.copy()

    if isinstance(image.reshape(-1)[0], np.float32):
        # Convert to 255 and np.uint8 for cv2..
        image = (image * 255).astype(np.uint8)

    kp = np.round(kp).astype(int)

    for kpi, color in zip(kp, colors):
        # This sometimes causes OverflowError,,
        if kpi[2] == 0:
            continue
        cv2.circle(image, (kpi[0], kpi[1]), radius + 1, white, -1)
        cv2.circle(image, (kpi[0], kpi[1]), radius, color, -1)

    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.clf()
    # plt.imshow(image)
    # import ipdb; ipdb.set_trace()
    return image


def vis_verts(mean_shape, verts, face, mvs=None, textures=None):
    """
    mean_shape: N x 3
    verts: B x N x 3
    face: numpy F x 3
    textures: B x F x T x T (x T) x 3
    """
    from psbody.mesh.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewers
    if mvs is None:
        mvs = MeshViewers((2, 3))

    num_row = len(mvs)
    num_col = len(mvs[0])

    mean_shape = convert2np(mean_shape)
    verts = convert2np(verts)

    num_show = min(num_row * num_col, verts.shape[0] + 1)

    mvs[0][0].set_dynamic_meshes([Mesh(mean_shape, face)])
    # 0th is mean shape:

    if textures is not None:
        tex = convert2np(textures)
    for k in np.arange(1, num_show):
        vert_here = verts[k - 1]
        if textures is not None:
            tex_here = tex[k - 1]
            fc = tex_here.reshape(tex_here.shape[0], -1, 3).mean(axis=1)
            mesh = Mesh(vert_here, face, fc=fc)
        else:
            mesh = Mesh(vert_here, face)
        mvs[int(k % num_row)][int(k / num_row)].set_dynamic_meshes([mesh])


def vis_vert2kp(verts, vert2kp, face, mvs=None):
    """
    verts: N x 3
    vert2kp: K x N

    For each keypoint, visualize its weights on each vertex.
    Base color is white, pick a color for each kp.
    Using the weights, interpolate between base and color.

    """
    from psbody.mesh.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewer, MeshViewers
    from psbody.mesh.sphere import Sphere

    num_kp = vert2kp.shape[0]
    if mvs is None:
        mvs = MeshViewers((4, 4))
    # mv = MeshViewer()
    # Generate colors
    import pylab
    cm = pylab.get_cmap('gist_rainbow')
    cms = 255 * np.array([cm(1. * i / num_kp)[:3] for i in range(num_kp)])
    base = np.zeros((1, 3)) * 255
    # base = np.ones((1, 3)) * 255

    verts = convert2np(verts)
    vert2kp = convert2np(vert2kp)

    num_row = len(mvs)
    num_col = len(mvs[0])

    colors = []
    for k in range(num_kp):
        # Nx1 for this kp.
        weights = vert2kp[k].reshape(-1, 1)
        # So we can see it,,
        weights = weights / weights.max()
        cm = cms[k, None]
        # Simple linear interpolation,,
        # cs = np.uint8((1-weights) * base + weights * cm)
        # In [0, 1]
        cs = ((1 - weights) * base + weights * cm) / 255.
        colors.append(cs)

        # sph = [Sphere(center=jc, radius=.03).to_mesh(c/255.) for jc, c in zip(vert,cs)]
        # mvs[int(k/4)][k%4].set_dynamic_meshes(sph)
        mvs[int(k % num_row)][int(k / num_row)].set_dynamic_meshes(
            [Mesh(verts, face, vc=cs)])


def tensor2im(image_tensor, imtype=np.uint8, scale_to_range_1=False):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if scale_to_range_1:
        image_numpy = image_numpy - np.min(image_numpy, axis=2, keepdims=True)
        image_numpy = image_numpy / np.max(image_numpy)
    else:
        # Clip to [0, 1]
        image_numpy = np.clip(image_numpy, 0, 1)

    return (image_numpy * 255).astype(imtype)
