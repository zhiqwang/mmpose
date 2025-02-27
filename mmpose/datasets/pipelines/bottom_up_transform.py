import cv2
import numpy as np

from mmpose.core.post_processing import (get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES
from .shared_transform import Compose


def _ceil_to_multiples_of(x, base=64):
    """Transform x to the integral multiple of the base."""
    return int(np.ceil(x / base)) * base


def _get_multi_scale_size(image,
                          input_size,
                          current_scale,
                          min_scale,
                          use_udp=False):
    """Get the size for multi-scale training.

    Args:
        image: Input image.
        input_size (int): Size of the image input.
        current_scale (float): Scale factor.
        min_scale (float): Minimal scale.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing multi-scale sizes.

        - (w_resized, h_resized) (tuple(int)): resized width/height
        - center (np.ndarray)image center
        - scale (np.ndarray): scales wrt width/height
    """
    h, w, _ = image.shape

    # calculate the size for min_scale
    min_input_size = _ceil_to_multiples_of(min_scale * input_size, 64)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            _ceil_to_multiples_of(min_input_size / w * h, 64) * current_scale /
            min_scale)
        if use_udp:
            scale_w = w - 1.0
            scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
        else:
            scale_w = w / 200.0
            scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            _ceil_to_multiples_of(min_input_size / h * w, 64) * current_scale /
            min_scale)
        if use_udp:
            scale_h = h - 1.0
            scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
        else:
            scale_h = h / 200.0
            scale_w = w_resized / h_resized * h / 200.0
    if use_udp:
        center = (scale_w / 2.0, scale_h / 2.0)
    else:
        center = np.array([round(w / 2.0), round(h / 2.0)])
    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def _resize_align_multi_scale(image, input_size, current_scale, min_scale):
    """Resize the images for multi-scale training.

    Args:
        image: Input image
        input_size (int): Size of the image input
        current_scale (float): Current scale
        min_scale (float): Minimal scale

    Returns:
        tuple: A tuple containing image info.

        - image_resized (tuple): size of resize image
        - center (np.ndarray): center of image
        - scale (np.ndarray): scale
    """
    size_resized, center, scale = _get_multi_scale_size(
        image, input_size, current_scale, min_scale)

    trans = get_affine_transform(center, scale, 0, size_resized)
    image_resized = cv2.warpAffine(image, trans, size_resized)

    return image_resized, center, scale


def _resize_align_multi_scale_udp(image, input_size, current_scale, min_scale):
    """Resize the images for multi-scale training.

    Args:
        image: Input image
        input_size (int): Size of the image input
        current_scale (float): Current scale
        min_scale (float): Minimal scale

    Returns:
        tuple: A tuple containing image info.

        - image_resized (tuple): size of resize image
        - center (np.ndarray): center of image
        - scale (np.ndarray): scale
    """
    size_resized, _, _ = _get_multi_scale_size(image, input_size,
                                               current_scale, min_scale, True)

    _, center, scale = _get_multi_scale_size(image, input_size, min_scale,
                                             min_scale, True)

    trans = get_warp_matrix(
        theta=0,
        size_input=np.array(scale, dtype=float),
        size_dst=np.array(size_resized, dtype=float) - 1.0,
        size_target=np.array(scale, dtype=float))
    image_resized = cv2.warpAffine(
        image.copy(), trans, size_resized, flags=cv2.INTER_LINEAR)

    return image_resized, center, scale


class HeatmapGenerator:
    """Generate heatmaps for bottom-up models.

    Args:
        num_joints (int): Number of keypoints
        output_size (int): Size of feature map
        sigma (int): Sigma of the heatmaps.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, output_size, num_joints, sigma=-1, use_udp=False):
        self.output_size = output_size
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_size / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        self.use_udp = use_udp
        if use_udp:
            self.x = np.arange(0, size, 1, np.float32)
            self.y = self.x[:, None]
        else:
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            x0, y0 = 3 * sigma + 1, 3 * sigma + 1
            self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, joints):
        """Generate heatmaps."""
        hms = np.zeros((self.num_joints, self.output_size, self.output_size),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_size or y >= self.output_size:
                        continue

                    if self.use_udp:
                        x0 = 3 * sigma + 1 + pt[0] - x
                        y0 = 3 * sigma + 1 + pt[1] - y
                        g = np.exp(-((self.x - x0)**2 + (self.y - y0)**2) /
                                   (2 * sigma**2))
                    else:
                        g = self.g

                    ul = int(np.round(x - 3 * sigma -
                                      1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma +
                                      2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_size) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_size) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_size)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_size)
                    hms[idx, aa:bb,
                        cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], g[a:b,
                                                                      c:d])
        return hms


class JointsEncoder:
    """Encodes the visible joints into (coordinates, score); The coordinate of
    one joint and its score are of `int` type.

    (idx * output_size**2 + y * output_size + x, 1) or (0, 0).

    Args:
        max_num_people(int): Max number of people in an image
        num_joints(int): Number of keypoints
        output_size(int): Size of feature map
        tag_per_joint(bool):  Option to use one tag map per joint.
    """

    def __init__(self, max_num_people, num_joints, output_size, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_size = output_size
        self.tag_per_joint = tag_per_joint

    def __call__(self, joints):
        """
        Note:
            number of people in image: N
            number of keypoints: K
            max number of people in an image: M

        Args:
            joints (np.ndarray[NxKx3])

        Returns:
            visible_kpts (np.ndarray[MxKx2]).
        """
        visible_kpts = np.zeros((self.max_num_people, self.num_joints, 2),
                                dtype=np.float32)
        output_size = self.output_size
        for i in range(len(joints)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if (pt[2] > 0 and 0 <= y < self.output_size
                        and 0 <= x < self.output_size):
                    if self.tag_per_joint:
                        visible_kpts[i][tot] = \
                            (idx * output_size**2 + y * output_size + x, 1)
                    else:
                        visible_kpts[i][tot] = (y * output_size + x, 1)
                    tot += 1
        return visible_kpts


class PAFGenerator:
    """Generate part affinity fields.

    Args:
        output_size (int): Size of feature map.
        limb_width (int): Limb width of part affinity fields.
        skeleton (list[list]): connections of joints.
    """

    def __init__(self, output_size, limb_width, skeleton):
        self.output_size = output_size
        self.limb_width = limb_width
        self.skeleton = skeleton

    def _accumulate_paf_map_(self, pafs, src, dst, count):
        """Accumulate part affinity fields between two given joints.

        Args:
            pafs (np.ndarray[2xHxW]): paf maps (2 dimensions:x axis and
                y axis) for a certain limb connection. This argument will
                be modified inplace.
            src (np.ndarray[2,]): coordinates of the source joint.
            dst (np.ndarray[2,]): coordinates of the destination joint.
            count (np.ndarray[HxW]): count map that preserves the number
                of non-zero vectors at each point. This argument will be
                modified inplace.
        """
        limb_vec = dst - src
        norm = np.linalg.norm(limb_vec)
        if norm == 0:
            unit_limb_vec = np.zeros(2)
        else:
            unit_limb_vec = limb_vec / norm

        min_x = max(np.floor(min(src[0], dst[0]) - self.limb_width), 0)
        max_x = min(
            np.ceil(max(src[0], dst[0]) + self.limb_width),
            self.output_size - 1)
        min_y = max(np.floor(min(src[1], dst[1]) - self.limb_width), 0)
        max_y = min(
            np.ceil(max(src[1], dst[1]) + self.limb_width),
            self.output_size + 1)

        range_x = list(range(int(min_x), int(max_x + 1), 1))
        range_y = list(range(int(min_y), int(max_y + 1), 1))
        xx, yy = np.meshgrid(range_x, range_y)
        delta_x = xx - src[0]
        delta_y = yy - src[1]
        dist = np.abs(delta_x * unit_limb_vec[1] - delta_y * unit_limb_vec[0])
        mask_local = (dist < self.limb_width)
        mask = np.zeros_like(count, dtype=bool)
        mask[xx, yy] = mask_local

        pafs[0, mask] += unit_limb_vec[0]
        pafs[1, mask] += unit_limb_vec[1]
        count += mask

        return pafs, count

    def __call__(self, joints):
        """Generate the target part affinity fields."""
        pafs = np.zeros(
            (len(self.skeleton) * 2, self.output_size, self.output_size),
            dtype=np.float32)

        for idx, sk in enumerate(self.skeleton):
            count = np.zeros((self.output_size, self.output_size),
                             dtype=np.float32)

            for p in joints:
                src = p[sk[0] - 1]
                dst = p[sk[1] - 1]
                if src[2] > 0 and dst[2] > 0:
                    self._accumulate_paf_map_(pafs[2 * idx:2 * idx + 2],
                                              src[:2], dst[:2], count)

            pafs[2 * idx:2 * idx + 2] /= np.maximum(count, 1)

        return pafs


@PIPELINES.register_module()
class BottomUpRandomFlip:
    """Data augmentation with random image flip for bottom-up.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        image, mask, joints = results['img'], results['mask'], results[
            'joints']
        self.flip_index = results['ann_info']['flip_index']
        self.output_size = results['ann_info']['heatmap_size']

        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if np.random.random() < self.flip_prob:
            image = image[:, ::-1] - np.zeros_like(image)
            for i, _output_size in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1]
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size - joints[i][:, :, 0] - 1
        results['img'], results['mask'], results[
            'joints'] = image, mask, joints
        return results


@PIPELINES.register_module()
class BottomUpRandomAffine:
    """Data augmentation with random scaling & rotating.

    Args:
        rot_factor (int): Rotating to [-rotation_factor, rotation_factor]
        scale_factor (float): Scaling to [1-scale_factor, 1+scale_factor]
        scale_type: wrt ``long`` or ``short`` length of the image.
        trans_factor: Translation factor.
        scale_aware_sigma: Option to use scale-aware sigma
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 rot_factor,
                 scale_factor,
                 scale_type,
                 trans_factor,
                 use_udp=False):
        self.max_rotation = rot_factor
        self.min_scale = scale_factor[0]
        self.max_scale = scale_factor[1]
        self.scale_type = scale_type
        self.trans_factor = trans_factor
        self.use_udp = use_udp

    @staticmethod
    def _get_affine_matrix(center, scale, res, rot=0):
        """Generate transformation matrix."""
        h = scale
        t = np.zeros((3, 3), dtype=np.float32)
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if rot != 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3), dtype=np.float32)
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        image, mask, joints = results['img'], results['mask'], results[
            'joints']

        self.input_size = results['ann_info']['image_size']
        self.output_size = results['ann_info']['heatmap_size']

        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size), (len(mask),
                                                    len(self.output_size),
                                                    self.output_size)

        height, width = image.shape[:2]
        if self.use_udp:
            center = np.array(((width - 1.0) / 2, (height - 1.0) / 2))
        else:
            center = np.array((width / 2, height / 2))
        if self.scale_type == 'long':
            scale = max(height, width) / 1.0
        elif self.scale_type == 'short':
            scale = min(height, width) / 1.0
        else:
            raise ValueError(f'Unknown scale type: {self.scale_type}')
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.trans_factor > 0:
            dx = np.random.randint(-self.trans_factor * scale / 200.0,
                                   self.trans_factor * scale / 200.0)
            dy = np.random.randint(-self.trans_factor * scale / 200.0,
                                   self.trans_factor * scale / 200.0)

            center[0] += dx
            center[1] += dy
        if self.use_udp:
            for i, _output_size in enumerate(self.output_size):
                trans = get_warp_matrix(
                    theta=aug_rot,
                    size_input=center * 2.0,
                    size_dst=np.array(
                        (_output_size, _output_size), dtype=float) - 1.0,
                    size_target=np.array((scale, scale), dtype=float))
                mask[i] = cv2.warpAffine(
                    (mask[i] * 255).astype(np.uint8),
                    trans, (int(_output_size), int(_output_size)),
                    flags=cv2.INTER_LINEAR) / 255
                mask[i] = (mask[i] > 0.5).astype(np.float32)
                joints[i][:, :, 0:2] = \
                    warp_affine_joints(joints[i][:, :, 0:2].copy(), trans)
                if results['ann_info']['scale_aware_sigma']:
                    joints[i][:, :, 3] = joints[i][:, :, 3] / aug_scale
            mat_input = get_warp_matrix(
                theta=aug_rot,
                size_input=center * 2.0,
                size_dst=np.array(
                    (self.input_size, self.input_size), dtype=float) - 1.0,
                size_target=np.array((scale, scale), dtype=float))
            image = cv2.warpAffine(
                image,
                mat_input, (int(self.input_size), int(self.input_size)),
                flags=cv2.INTER_LINEAR)
        else:
            for i, _output_size in enumerate(self.output_size):
                mat_output = self._get_affine_matrix(center, scale,
                                                     (_output_size,
                                                      _output_size),
                                                     aug_rot)[:2]
                mask[i] = cv2.warpAffine(
                    (mask[i] * 255).astype(np.uint8), mat_output,
                    (_output_size, _output_size)) / 255
                mask[i] = (mask[i] > 0.5).astype(np.float32)

                joints[i][:, :, 0:2] = \
                    warp_affine_joints(joints[i][:, :, 0:2], mat_output)
                if results['ann_info']['scale_aware_sigma']:
                    joints[i][:, :, 3] = joints[i][:, :, 3] / aug_scale
            mat_input = self._get_affine_matrix(center, scale,
                                                (self.input_size,
                                                 self.input_size), aug_rot)[:2]
            image = cv2.warpAffine(image, mat_input, (self.input_size.item(),
                                                      self.input_size.item()))

        results['img'], results['mask'], results[
            'joints'] = image, mask, joints

        return results


@PIPELINES.register_module()
class BottomUpGenerateHeatmapTarget:
    """Generate multi-scale heatmap target for bottom-up.

    Args:
        sigma (int): Sigma of heatmap Gaussian
        max_num_people (int): Maximum number of people in an image
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, sigma, use_udp=False):
        self.sigma = sigma
        self.use_udp = use_udp

    def _generate(self, num_joints, heatmap_size):
        """Get heatmap generator."""
        heatmap_generator = [
            HeatmapGenerator(output_size, num_joints, self.sigma, self.use_udp)
            for output_size in heatmap_size
        ]
        return heatmap_generator

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        heatmap_generator = \
            self._generate(results['ann_info']['num_joints'],
                           results['ann_info']['heatmap_size'])
        target_list = list()
        joints_list = results['joints']

        for scale_id in range(results['ann_info']['num_scales']):
            heatmaps = heatmap_generator[scale_id](joints_list[scale_id])
            target_list.append(heatmaps.astype(np.float32))
        results['target'] = target_list

        return results


@PIPELINES.register_module()
class BottomUpGenerateTarget:
    """Generate multi-scale heatmap target for bottom-up.

    Args:
        sigma (int): Sigma of heatmap Gaussian
        max_num_people (int): Maximum number of people in an image
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, sigma, max_num_people, use_udp=False):
        self.sigma = sigma
        self.max_num_people = max_num_people
        self.use_udp = use_udp

    def _generate(self, num_joints, heatmap_size):
        """Get heatmap generator and joint encoder."""
        heatmap_generator = [
            HeatmapGenerator(output_size, num_joints, self.sigma, self.use_udp)
            for output_size in heatmap_size
        ]
        joints_encoder = [
            JointsEncoder(self.max_num_people, num_joints, output_size, True)
            for output_size in heatmap_size
        ]
        return heatmap_generator, joints_encoder

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        heatmap_generator, joints_encoder = \
            self._generate(results['ann_info']['num_joints'],
                           results['ann_info']['heatmap_size'])
        target_list = list()
        img, mask_list, joints_list = results['img'], results['mask'], results[
            'joints']

        for scale_id in range(results['ann_info']['num_scales']):
            target_t = heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = joints_encoder[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        results['img'], results['masks'], results[
            'joints'] = img, mask_list, joints_list
        results['targets'] = target_list

        return results


@PIPELINES.register_module()
class BottomUpGeneratePAFTarget:
    """Generate multi-scale heatmaps and part affinity fields (PAF) target for
    bottom-up. Paper ref: Cao et al. Realtime Multi-Person 2D Human Pose
    Estimation using Part Affinity Fields (CVPR 2017).

    Args:
        limb_width (int): Limb width of part affinity fields
    """

    def __init__(self, limb_width, skeleton=None):
        self.limb_width = limb_width
        self.skeleton = skeleton

    def _generate(self, heatmap_size, skeleton):
        """Get PAF generator."""
        paf_generator = [
            PAFGenerator(output_size, self.limb_width, skeleton)
            for output_size in heatmap_size
        ]
        return paf_generator

    def __call__(self, results):
        """Generate multi-scale part affinity fields for bottom-up."""
        if self.skeleton is None:
            assert results['ann_info']['skeleton'] is not None
            self.skeleton = results['ann_info']['skeleton']
        else:
            assert np.array(
                self.skeleton).max() < results['ann_info']['num_joints']

        paf_generator = \
            self._generate(results['ann_info']['heatmap_size'],
                           self.skeleton)
        target_list = list()
        joints_list = results['joints']

        for scale_id in range(results['ann_info']['num_scales']):
            pafs = paf_generator[scale_id](joints_list[scale_id])
            target_list.append(pafs.astype(np.float32))

        results['target'] = target_list

        return results


@PIPELINES.register_module()
class BottomUpGetImgSize:
    """Get multi-scale image sizes for bottom-up, including base_size and
    test_scale_factor. Keep the ratio and the image is resized to
    `results['ann_info']['image_size']×current_scale`.

    Args:
        test_scale_factor (List[float]): Multi scale
        current_scale (int): default 1
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, test_scale_factor, current_scale=1, use_udp=False):
        self.test_scale_factor = test_scale_factor
        self.min_scale = min(test_scale_factor)
        self.current_scale = current_scale
        self.use_udp = use_udp

    def __call__(self, results):
        """Get multi-scale image sizes for bottom-up."""
        input_size = results['ann_info']['image_size']
        img = results['img']

        h, w, _ = img.shape

        # calculate the size for min_scale
        min_input_size = _ceil_to_multiples_of(self.min_scale * input_size, 64)
        if w < h:
            w_resized = int(min_input_size * self.current_scale /
                            self.min_scale)
            h_resized = int(
                _ceil_to_multiples_of(min_input_size / w * h, 64) *
                self.current_scale / self.min_scale)
            if self.use_udp:
                scale_w = w - 1.0
                scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
            else:
                scale_w = w / 200.0
                scale_h = h_resized / w_resized * w / 200.0
        else:
            h_resized = int(min_input_size * self.current_scale /
                            self.min_scale)
            w_resized = int(
                _ceil_to_multiples_of(min_input_size / h * w, 64) *
                self.current_scale / self.min_scale)
            if self.use_udp:
                scale_h = h - 1.0
                scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
            else:
                scale_h = h / 200.0
                scale_w = w_resized / h_resized * h / 200.0
        if self.use_udp:
            center = (scale_w / 2.0, scale_h / 2.0)
        else:
            center = np.array([round(w / 2.0), round(h / 2.0)])
        results['ann_info']['test_scale_factor'] = self.test_scale_factor
        results['ann_info']['base_size'] = (w_resized, h_resized)
        results['ann_info']['center'] = center
        results['ann_info']['scale'] = np.array([scale_w, scale_h])

        return results


@PIPELINES.register_module()
class BottomUpResizeAlign:
    """Resize multi-scale size and align transform for bottom-up.

    Args:
        transforms (List): ToTensor & Normalize
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, transforms, use_udp=False):
        self.transforms = Compose(transforms)
        if use_udp:
            self._resize_align_multi_scale = _resize_align_multi_scale_udp
        else:
            self._resize_align_multi_scale = _resize_align_multi_scale

    def __call__(self, results):
        """Resize multi-scale size and align transform for bottom-up."""
        input_size = results['ann_info']['image_size']
        test_scale_factor = results['ann_info']['test_scale_factor']
        aug_data = []

        for _, s in enumerate(sorted(test_scale_factor, reverse=True)):
            _results = results.copy()
            image_resized, _, _ = self._resize_align_multi_scale(
                _results['img'], input_size, s, min(test_scale_factor))
            _results['img'] = image_resized
            _results = self.transforms(_results)
            transformed_img = _results['img'].unsqueeze(0)
            aug_data.append(transformed_img)

        results['ann_info']['aug_data'] = aug_data

        return results
