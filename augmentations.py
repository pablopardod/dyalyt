import random
from albumentations.core.transforms_interface import BasicTransform
from albumentations import Compose
from torch.nn import functional as F
import torch
import numpy as np
import math
import typing
import pdb

# CutMix
from torchvision.transforms.v2._augment import CutMix, query_size 
from typing import Any, Callable, Dict, List, Tuple

def crop_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # Crop
        if mode == "start":
            data = data[:max_len]
        else:
            offset = np.abs(diff) // 2
            data = data[offset: offset + max_len]
        return data
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2])) * coef
    data = torch.cat([data, padding])
    return data

          
class Resample(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(0.8,1.2),
        always_apply=False,
        p=0.5,
    ):
        super(Resample, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, sample_rate=1., **params):
        length = data.shape[0]
        new_size = max(int(length * sample_rate),15)
        new_x = F.interpolate(data.permute(1,2,0),new_size).permute(2,0,1)
        return new_x

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}


    
class TemporalMask(BasicTransform):
    """
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        mask_value=float('nan'),
        num_masks = (1,2),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalMask, self).__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(self, data, mask_sizes=[0.3],mask_offsets_01=[0.2], mask_value=float('nan'), **params):
        l = data.shape[0]
        x_new = data.clone()
        for mask_size, mask_offset_01 in zip(mask_sizes,mask_offsets_01):
            mask_size = int(l * mask_size)
            max_mask = np.clip(l-mask_size,1,l)
            mask_offset = int(mask_offset_01 * max_mask)
            x_new[mask_offset:mask_offset+mask_size] = torch.tensor(mask_value)
        return x_new

    def get_params(self):
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_size = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {"mask_sizes": mask_size,
                'mask_offsets_01':mask_offset_01,
                'mask_value':self.mask_value,}

    def get_transform_init_args_names(self):
        return ("size","mask_value","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
    
class SpatialMask(BasicTransform):
    """    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.5,1.), 
        mask_value=float('nan'),
        mode = 'abolute',
        always_apply=False,
        p=0.5,
    ):
        super(SpatialMask, self).__init__(always_apply, p)

        self.size = size
        self.mask_value = mask_value
        self.mode = mode

    def apply(self, data, mask_size=0.75, offset_x_01=0.2, offset_y_01=0.2,mask_value=float('nan'), **params):
        # mask_size absolute width 
        
        
        
        #fill na makes it easier with min and max
        data0 = data.contiguous()
        data0[torch.isnan(data0)] = 0
        
        x_min, x_max = data0[...,0].min().item(), data0[...,0].max().item() 
        y_min, y_max = data0[...,1].min().item(), data0[...,1].max().item() 
        
        if self.mode == 'relative':
            mask_size_x = mask_size * (x_max - x_min)
            mask_size_y = mask_size * (y_max - y_min)
        else:
            mask_size_x = mask_size 
            mask_size_y = mask_size             

        mask_offset_x = offset_x_01 * (x_max - x_min) + x_min
        mask_offset_y = offset_y_01 * (y_max - y_min) + y_min
        
        mask_x = (mask_offset_x<data0[...,0]) & (data0[...,0] < mask_offset_x + mask_size_x)
        mask_y = (mask_offset_y<data0[...,1]) & (data0[...,1] < mask_offset_y + mask_size_y)
        
        mask = mask_x & mask_y
        x_new = data.contiguous() * (1-mask[:,:,None].float()) + mask[:,:,None] * mask_value
        return data

    def get_params(self):
        params = {"offset_x_01": random.uniform(0, 1)}
        params['offset_y_01'] = random.uniform(0, 1)
        params['mask_size'] = random.uniform(self.size[0], self.size[1])
        params['mask_value'] = self.mask_value
        return params

    def get_transform_init_args_names(self):
        return ("size", "mask_value","mode")
    
    @property
    def targets(self):
        return {"image": self.apply}  

     
    
class TimeShift(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        shift_rate=(-10,10),
        always_apply=False,
        p=0.5,
    ):
        super(TimeShift, self).__init__(always_apply, p)
        
        rate_lower = shift_rate[0]
        rate_upper = shift_rate[1]

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, shift_rate=5, **params):
        length = data.shape[0]
        
        if shift_rate > 0:
            zeros = torch.zeros((shift_rate,data.shape[1],data.shape[2]),dtype=data.dtype)
            new_x = torch.cat([zeros,data.clone()])
        elif shift_rate > -data.shape[0]+2:
            new_x = data.clone()[-shift_rate:]
        else:
            new_x = data.clone()

        return new_x

    def get_params(self):
        return {"shift_rate": np.random.randint(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}

    
class FaceDrop(BasicTransform):
    """    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FaceDrop, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if '_x' in i]
        self.face_indices = np.array([t for t,l in enumerate(landmarks) if l in ['nose_x','neck_x','lEye_x','rEye_x','lEar_x','rEar_x']])
        self.mask_value = mask_value

    def apply(self, data,pidx=None, **params):
        x_new = data.contiguous()
        
        drop_indices = self.face_indices[pidx].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        pidx = range(len(self.face_indices)) #all pose idxs
        params = {'pidx':pidx}
        return params

    def get_transform_init_args_names(self):
        return ("mask_value",)
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
    
class OneOf(BasicTransform):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, 
                always_apply=False,
                p=1.0,):
        super(OneOf, self).__init__(always_apply, p)
        self.transforms = transforms
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:

        if self.transforms_ps and random.random() < self.p:
            idx: int = np.random.choice(range(len(self.transforms)), p=self.transforms_ps, size = 1)[0]
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data
    
    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()
    
    @property
    def targets(self):
        return {"image": self.apply}  

    
class DynamicResample(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(0.8,1.2),
        windows = (5,10),
        always_apply=False,
        p=0.5,
    ):
        super(DynamicResample, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))
        
        self.rate_lower = rate_lower
        self.rate_upper = rate_upper
        self.windows = windows

    def apply(self, data, sample_rates=[1.], **params):
        
        sample_rates = sample_rates[:data.shape[0]] #handle very short seq e.g. seq_len=6 
        
        chunks = data.chunk(len(sample_rates))
        new_x = []
        for sample_rate, chunk in zip(sample_rates,chunks):
            length = chunk.shape[0]
            new_size = max(int(length * sample_rate),1)
            new_x += [F.interpolate(chunk.permute(1,2,0),new_size).permute(2,0,1)]
        new_x = torch.cat(new_x)
        return new_x

    def get_params(self):
        w = np.random.randint(self.windows[0],self.windows[1])
        sample_rates = [random.uniform(self.rate_lower, self.rate_upper) for _ in range(w)]
        return {"sample_rates": sample_rates,}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper","windows")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
    
class PoseDrop2(BasicTransform):
    """    
    Args:
        landmarks : xyz_landmarks .. array of strings
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(PoseDrop2, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if '_x' in i]

        pose_indices = np.array([t for t,l in enumerate(landmarks) if l not in ['nose_x','neck_x','lEye_x','rEye_x','lEar_x','rEar_x']])[:-1]
        pose_indices = pose_indices.reshape(-1, 2).T
        self.pose_indices_type1 = pose_indices[:,0:].T.reshape(-1)
        self.pose_indices_type2 = pose_indices[:,1:].T.reshape(-1)
        self.pose_indices_type3 = pose_indices[:,2:].T.reshape(-1)
        self.pose_indices_type4 = pose_indices[:,3:].T.reshape(-1)
                
        self.mask_value = mask_value

    def apply(self, data, **params):
        x_new = data.contiguous()
        
        pose_indices = random.choice([self.pose_indices_type1,
                                      self.pose_indices_type2,
                                      self.pose_indices_type3,
                                      self.pose_indices_type4])
        drop_indices = pose_indices.flatten()

        #Editado del original
        slice_shape = (x_new.shape[0], len(drop_indices), 3)
        mask_tensor = torch.full(slice_shape, self.mask_value, dtype=x_new.dtype)
        x_new[:, drop_indices] = mask_tensor
        
        return x_new

    def get_params(self):
        #pidx = range(len(self.pose_indices)) #all pose idxs
        #params = {'pidx':pidx}
        params = {}
        return params

    def get_transform_init_args_names(self):
        return ( "mask_value",)
    
    @property
    def targets(self):
        return {"image": self.apply}  

class CutVideo(BasicTransform):
    """
    Cuts the video from both the front and the back
    
    Args:
        rate (int,int): maximum front and end amount of frames to cut. Should both be int

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        limits=(10,10),
        always_apply=False,
    ):
        super(CutVideo, self).__init__(always_apply)

        self.start = limits[0]
        self.end = limits[1]

    def apply(self, data, random=(1.,1.), **params):
        length = data.shape[0]
        start = int(self.start * random[0])
        end = int(self.end * random[1])

        new_length = length - start - end

        if new_length <= 0:
            return data
        
        new_x = data[start:new_length+start]
        return new_x

    def get_params(self):
        random_number1 = random.random()
        random_number2 = random.random()
        return {"random": (random_number1,random_number2)}

    def get_transform_init_args_names(self):
        return ("start", "end")
    
    @property
    def targets(self):
        return {"image": self.apply}


class HeightPerson(BasicTransform):
    """
    stretches/ squeezes height of person
    
    Args:
        rate (float,float): lower and upper height of the person.

    Targets:
        pose

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(0.5,1.3),
        always_apply=False,
        p=0.5,
    ):
        super(HeightPerson, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, sample_rate=1., **params):
        # Multiply the y of each landmark by the sample rate
        # Data format: seq_len, n_landmarks, 3
        new_x = data.clone()
        new_x[...,1] = new_x[...,1] * sample_rate       
        return new_x

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}
    

class WidthPerson(BasicTransform):
    """
    stretches/ squeezes width of person
    
    Args:
        rate (float,float): lower and upper width of the person.

    Targets:
        pose

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(0.5,1.3),
        always_apply=False,
        p=0.5,
    ):
        super(WidthPerson, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, sample_rate=1., **params):
        # Multiply the y of each landmark by the sample rate
        # Data format: seq_len, n_landmarks, 3
        new_x = data.clone()
        new_x[...,0] = new_x[...,0] * sample_rate       
        return new_x

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}
    

class RotatePerson(BasicTransform):
    """
    rotates position of the person
    
    Args:
        rate (float,float): negative and positive rotation of the person.

    Targets:
        pose

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(-12,12),
        always_apply=False,
        p=0.5,
    ):
        super(RotatePerson, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def rotate_y_axis(self, landmarks, angle_degrees):
        # landmarks: torch.Tensor with shape (num_frames, num_landmarks, 3)
        # angle_degrees: rotation angle in degrees

        # Convert the angle to radians
        angle_radians = torch.deg2rad(torch.tensor(angle_degrees))

        # Rotation matrix around the y-axis
        rotation_matrix = torch.tensor([
            [torch.cos(angle_radians), 0, torch.sin(angle_radians)],
            [0, 1, 0],
            [-torch.sin(angle_radians), 0, torch.cos(angle_radians)]
        ])

        # Apply the rotation to each landmark
        rotated_landmarks = torch.matmul(landmarks, rotation_matrix.T)

        return rotated_landmarks

    def apply(self, data, sample_rate=1., **params):
        # Multiply the y of each landmark by the sample rate
        # Data format: seq_len, n_landmarks, 3
        rotated_landmarks = self.rotate_y_axis(data, sample_rate)    
        return rotated_landmarks
    
    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}


class CutMixModified(CutMix):
    """Apply CutMix to the provided batch of images and labels.

    Paper: `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    <https://arxiv.org/abs/1905.04899>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``CutMix()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """
    def __init__(self, num_classes, mode):   
        super().__init__(num_classes=num_classes)
        self.mode = mode


    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H, W = query_size(flat_inputs)

        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        
        if self.mode == 'horizontal':
            y1 = 0
            y2 = H
        elif self.mode == 'vertical':
            x1 = 0
            x2 = W

        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        return dict(box=box, lam_adjusted=lam_adjusted)
