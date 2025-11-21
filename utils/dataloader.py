# # # Copyright by HQ-SAM team
# # # All rights reserved.

# # ## data loader
# # from __future__ import print_function, division

# # import numpy as np
# # import random
# # from copy import deepcopy
# # from skimage import io
# # import os
# # from glob import glob

# # import torch
# # from torch.utils.data import Dataset, DataLoader, ConcatDataset
# # from torchvision import transforms, utils
# # from torchvision.transforms.functional import normalize
# # import torch.nn.functional as F
# # from torch.utils.data.distributed import DistributedSampler

# # #### --------------------- dataloader online ---------------------####

# # def get_im_gt_name_dict(datasets, flag='valid'):
# #     print("------------------------------", flag, "--------------------------------")
# #     name_im_gt_list = []

# #     for i in range(len(datasets)):
# #         print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
# #         tmp_im_list, tmp_gt_list = [], []
# #         tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
# #         print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
# #         tmp_im_ch4_list, tmp_im_ch5_list = [],[]
# #         tmp_im_ch4_list = glob(datasets[i]["im_ch4_dir"]+os.sep+'*'+datasets[i]["im_ext"])
# #         print('-im ch4-',datasets[i]["name"],datasets[i]["im_ch4_dir"], ': ',len(tmp_im_ch4_list))
        
# #         tmp_im_ch5_list = glob(datasets[i]["im_ch5_dir"]+os.sep+'*'+datasets[i]["im_ext"])
# #         print('-im ch5-',datasets[i]["name"],datasets[i]["im_ch5_dir"], ': ',len(tmp_im_ch5_list))

# #         if(datasets[i]["gt_dir"]==""):
# #             print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
# #             tmp_gt_list = []
# #         else:
# #             tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
# #             print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


# #         name_im_gt_list.append({"dataset_name":datasets[i]["name"],
# #                                 "im_path":tmp_im_list,
# #                                 "gt_path":tmp_gt_list,
# #                                 "im_ext":datasets[i]["im_ext"],
# #                                 "gt_ext":datasets[i]["gt_ext"],
# #                                 "im_ch4_path":tmp_im_ch4_list,
# #                                 "im_ch5_path":tmp_im_ch5_list
# #         })
                               

# #     return name_im_gt_list

# # def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
# #     gos_dataloaders = []
# #     gos_datasets = []

# #     if(len(name_im_gt_list)==0):
# #         return gos_dataloaders, gos_datasets

# #     num_workers_ = 1
# #     if(batch_size>1):
# #         num_workers_ = 2
# #     if(batch_size>4):
# #         num_workers_ = 4
# #     if(batch_size>8):
# #         num_workers_ = 8


# #     if training:
# #         for i in range(len(name_im_gt_list)):   
# #             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
# #             gos_datasets.append(gos_dataset)

# #         gos_dataset = ConcatDataset(gos_datasets)
# #         sampler = DistributedSampler(gos_dataset)
# #         batch_sampler_train = torch.utils.data.BatchSampler(
# #             sampler, batch_size, drop_last=True)
# #         dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

# #         gos_dataloaders = dataloader
# #         gos_datasets = gos_dataset

# #     else:
# #         for i in range(len(name_im_gt_list)):   
# #             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
# #             sampler = DistributedSampler(gos_dataset, shuffle=False)
# #             dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

# #             gos_dataloaders.append(dataloader)
# #             gos_datasets.append(gos_dataset)

# #     return gos_dataloaders, gos_datasets

# # class RandomHFlip(object):
# #     def __init__(self,prob=0.5):
# #         self.prob = prob
# #     def __call__(self,sample):
# #         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
# #         im_5channel= sample['image_5channel']

# #         # random horizontal flip
# #         if random.random() >= self.prob:
# #             image = torch.flip(image,dims=[2])
# #             label = torch.flip(label,dims=[2])
# #             im_5channel= torch.flip(im_5channel,dims=[2])

# #         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape, 'image_5channel':im_5channel}

# # class Resize(object):
# #     def __init__(self,size=[320,320]):
# #         self.size = size
# #     def __call__(self,sample):
# #         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
# #         im_5channel= sample['image_5channel']

# #         image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
# #         label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)
# #         im_5channel = torch.squeeze(F.interpolate(torch.unsqueeze(im_5channel,0),self.size,mode='bilinear'),dim=0)

# #         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size), 'image_5channel':im_5channel}

# # class RandomCrop(object):
# #     def __init__(self,size=[288,288]):
# #         self.size = size
# #     def __call__(self,sample):
# #         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

# #         h, w = image.shape[1:]
# #         new_h, new_w = self.size

# #         top = np.random.randint(0, h - new_h)
# #         left = np.random.randint(0, w - new_w)

# #         image = image[:,top:top+new_h,left:left+new_w]
# #         label = label[:,top:top+new_h,left:left+new_w]

# #         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


# # class Normalize(object):
# #     def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
# #         self.mean = mean
# #         self.std = std

# #     def __call__(self,sample):

# #         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
# #         image = normalize(image,self.mean,self.std)

# #         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



# # class LargeScaleJitter(object):
# #     """
# #         implementation of large scale jitter from copy_paste
# #         https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
# #     """

# #     def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
# #         self.desired_size = torch.tensor(output_size)
# #         self.aug_scale_min = aug_scale_min
# #         self.aug_scale_max = aug_scale_max

# #     def pad_target(self, padding, target):
# #         target = target.copy()
# #         if "masks" in target:
# #             target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
# #         return target

# #     def __call__(self, sample):
# #         imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']
# #         im_5channel= sample['image_5channel']

# #         #resize keep ratio
# #         out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

# #         random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
# #         scaled_size = (random_scale * self.desired_size).round()

# #         scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
# #         scaled_size = (image_size * scale).round().long()
        
# #         scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
# #         scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
# #         scaled_im_5channel = torch.squeeze(F.interpolate(torch.unsqueeze(im_5channel,0),scaled_size.tolist(),mode='bilinear'),dim=0)
# #         # random crop
# #         crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

# #         margin_h = max(scaled_size[0] - crop_size[0], 0).item()
# #         margin_w = max(scaled_size[1] - crop_size[1], 0).item()
# #         offset_h = np.random.randint(0, margin_h + 1)
# #         offset_w = np.random.randint(0, margin_w + 1)
# #         crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
# #         crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

# #         scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
# #         scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]
# #         scaled_im_5channel = scaled_im_5channel[:,crop_y1:crop_y2, crop_x1:crop_x2]

# #         # pad
# #         padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
# #         padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
# #         image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
# #         label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

# #         im_5channel = F.pad(scaled_im_5channel, [0,padding_w, 0,padding_h],value=128)
        
# #         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:]), 'image_5channel':im_5channel}






# # class OnlineDataset(Dataset):
# #     def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

# #         self.transform = transform
# #         self.dataset = {}
# #         ## combine different datasets into one
# #         dataset_names = []
# #         dt_name_list = [] # dataset name per image
# #         im_name_list = [] # image name
# #         im_path_list = [] # im path
# #         gt_path_list = [] # gt path
# #         im_ext_list = [] # im ext
# #         gt_ext_list = [] # gt ext
# #         im_ch4_name_list = []
# #         im_ch5_name_list = []
# #         im_ch4_path_list = []
# #         im_ch5_path_list = []
        
# #         for i in range(0,len(name_im_gt_list)):
# #             dataset_names.append(name_im_gt_list[i]["dataset_name"])
# #             # dataset name repeated based on the number of images in this dataset
# #             dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
# #             im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
# #             im_path_list.extend(name_im_gt_list[i]["im_path"])
# #             gt_path_list.extend(name_im_gt_list[i]["gt_path"])
# #             im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
# #             gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])
            
# #             im_ch4_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_ch4_path"]])
# #             im_ch5_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_ch5_path"]])
# #             im_ch4_path_list.extend(name_im_gt_list[i]["im_ch4_path"])
# #             im_ch5_path_list.extend(name_im_gt_list[i]["im_ch5_path"])


# #         self.dataset["data_name"] = dt_name_list
# #         self.dataset["im_name"] = im_name_list
# #         self.dataset["im_path"] = im_path_list
# #         self.dataset["ori_im_path"] = deepcopy(im_path_list)
# #         self.dataset["gt_path"] = gt_path_list
# #         self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
# #         self.dataset["im_ext"] = im_ext_list
# #         self.dataset["gt_ext"] = gt_ext_list
        
# #         self.dataset["im_ch4_name"] = im_ch4_name_list
# #         self.dataset["im_ch5_name"] = im_ch5_name_list
        
# #         self.dataset["im_ch4_path"] = im_ch4_path_list
# #         self.dataset["im_ch5_path"] = im_ch5_path_list

# #         self.eval_ori_resolution = eval_ori_resolution

# #     def __len__(self):
# #         return len(self.dataset["im_path"])
# #     def __getitem__(self, idx):
# #         im_path = self.dataset["im_path"][idx]
# #         gt_path = self.dataset["gt_path"][idx]
# #         im = io.imread(im_path)
# #         gt = io.imread(gt_path)
# #         # original code 10/01
# #         im_ch4_path = self.dataset["im_ch4_path"][idx]
# #         im_ch4 = io.imread(im_ch4_path)
        
# #         im_ch5_path = self.dataset["im_ch5_path"][idx]
# #         im_ch5 = io.imread(im_ch5_path)
        

# #         if len(gt.shape) > 2:
# #             gt = gt[:, :, 0]
# #         if len(im.shape) < 3:
# #             im = im[:, :, np.newaxis]
# #         if im.shape[2] == 1:
# #             im = np.repeat(im, 3, axis=2)
        
# #         #added below 2 lines HaoLan
# #         if im_ch4.ndim == 2:
# #             im_ch4 = im_ch4[:, :, np.newaxis]
# #         if im_ch4.shape[2] > 1:
# #             im_ch4 = im_ch4[:,:,:1]  
            
# #         #added below 2 lines HaoLan
# #         if im_ch5.ndim == 2:
# #             im_ch5 = im_ch5[:, :, np.newaxis]
# #         if im_ch5.shape[2] > 1:
# #             im_ch5 = im_ch5[:,:,:1]  
# #         #print(f'image shape: {im.shape}; ch4 shape: {im_ch4.shape}; ch5 shape: {im_ch5.shape} #################################################3')

# #         # Create two additional grayscale channels, TODO: change to images
# #         # height, width = im.shape[:2]
# #         # gray1 = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)
# #         # gray2 = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)

# #         # Concatenate the original image with the two new grayscale channels
# #         #im_5channel = np.concatenate((im, gray1, gray2), axis=2)
# #         im_5channel = np.concatenate((im, im_ch4, im_ch5), axis=2)
# #         # Convert to PyTorch tensor and rearrange dimensions
# #         im_5channel = torch.tensor(im_5channel.copy(), dtype=torch.float32)
# #         im_5channel = torch.permute(im_5channel, (2, 0, 1))

# #         im = torch.tensor(im.copy(), dtype=torch.float32)
# #         im = torch.transpose(torch.transpose(im,1,2),0,1)

# #         gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)
        
# #         #print(f'im_5channel shape: {im_5channel.shape} aaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
# #         im_5channel = im_5channel[:4,:,:]    
# #         #print(f'im_5channel shape: {im_5channel.shape} bbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        
# #         sample = {
# #         "imidx": torch.from_numpy(np.array(idx)),
# #         "image": im,
# #         "label": gt,
# #         "shape": torch.tensor(im.shape[-2:]),
# #         "image_5channel": im_5channel
# #         }
        
# #         if self.transform:
# #             sample = self.transform(sample)

# #         if self.eval_ori_resolution:
# #             sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
# #             sample['ori_im_path'] = self.dataset["im_path"][idx]
# #             sample['ori_gt_path'] = self.dataset["gt_path"][idx]

# #         return sample

# # ----BELOW is the original dataloader, for reference
# # # Copyright by HQ-SAM team
# # # All rights reserved.

# from __future__ import print_function, division

# import numpy as np
# import random
# from copy import deepcopy
# from skimage import io
# import os
# from glob import glob

# import torch
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torchvision import transforms, utils
# from torchvision.transforms.functional import normalize
# import torch.nn.functional as F
# from torch.utils.data.distributed import DistributedSampler

# #### --------------------- dataloader online ---------------------####

# def get_im_gt_name_dict(datasets, flag='valid'):
#     print("------------------------------", flag, "--------------------------------")
#     name_im_gt_list = []

#     for i in range(len(datasets)):
#         print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
#         tmp_im_list, tmp_gt_list = [], []
#         tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
#         print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

#         if(datasets[i]["gt_dir"]==""):
#             print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
#             tmp_gt_list = []
#         else:
#             tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
#             print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


#         name_im_gt_list.append({"dataset_name":datasets[i]["name"],
#                                 "im_path":tmp_im_list,
#                                 "gt_path":tmp_gt_list,
#                                 "im_ext":datasets[i]["im_ext"],
#                                 "gt_ext":datasets[i]["gt_ext"]})

#     return name_im_gt_list

# def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
#     gos_dataloaders = []
#     gos_datasets = []

#     if(len(name_im_gt_list)==0):
#         return gos_dataloaders, gos_datasets

#     num_workers_ = 1
#     if(batch_size>1):
#         num_workers_ = 2
#     if(batch_size>4):
#         num_workers_ = 4
#     if(batch_size>8):
#         num_workers_ = 8


#     if training:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
#             gos_datasets.append(gos_dataset)

#         gos_dataset = ConcatDataset(gos_datasets)
#         sampler = DistributedSampler(gos_dataset)
#         batch_sampler_train = torch.utils.data.BatchSampler(
#             sampler, batch_size, drop_last=True)
#         dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

#         gos_dataloaders = dataloader
#         gos_datasets = gos_dataset

#     else:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
#             sampler = DistributedSampler(gos_dataset, shuffle=False)
#             dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

#             gos_dataloaders.append(dataloader)
#             gos_datasets.append(gos_dataset)

#     return gos_dataloaders, gos_datasets

# class RandomHFlip(object):
#     def __init__(self,prob=0.5):
#         self.prob = prob
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         # random horizontal flip
#         if random.random() >= self.prob:
#             image = torch.flip(image,dims=[2])
#             label = torch.flip(label,dims=[2])

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

# class Resize(object):
#     def __init__(self,size=[320,320]):
#         self.size = size
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
#         label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}

# class RandomCrop(object):
#     def __init__(self,size=[288,288]):
#         self.size = size
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         h, w = image.shape[1:]
#         new_h, new_w = self.size

#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         image = image[:,top:top+new_h,left:left+new_w]
#         label = label[:,top:top+new_h,left:left+new_w]

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


# class Normalize(object):
#     def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
#         self.mean = mean
#         self.std = std

#     def __call__(self,sample):

#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
#         image = normalize(image,self.mean,self.std)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



# class LargeScaleJitter(object):
#     """
#         implementation of large scale jitter from copy_paste
#         https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
#     """

#     def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
#         self.desired_size = torch.tensor(output_size)
#         self.aug_scale_min = aug_scale_min
#         self.aug_scale_max = aug_scale_max

#     def pad_target(self, padding, target):
#         target = target.copy()
#         if "masks" in target:
#             target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
#         return target

#     def __call__(self, sample):
#         imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         #resize keep ratio
#         out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

#         random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
#         scaled_size = (random_scale * self.desired_size).round()

#         scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
#         scaled_size = (image_size * scale).round().long()
        
#         scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
#         scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
#         # random crop
#         crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

#         margin_h = max(scaled_size[0] - crop_size[0], 0).item()
#         margin_w = max(scaled_size[1] - crop_size[1], 0).item()
#         offset_h = np.random.randint(0, margin_h + 1)
#         offset_w = np.random.randint(0, margin_w + 1)
#         crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
#         crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

#         scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
#         scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

#         # pad
#         padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
#         padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
#         image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
#         label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}






# class OnlineDataset(Dataset):
#     def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

#         self.transform = transform
#         self.dataset = {}
#         ## combine different datasets into one
#         dataset_names = []
#         dt_name_list = [] # dataset name per image
#         im_name_list = [] # image name
#         im_path_list = [] # im path
#         gt_path_list = [] # gt path
#         im_ext_list = [] # im ext
#         gt_ext_list = [] # gt ext
#         for i in range(0,len(name_im_gt_list)):
#             dataset_names.append(name_im_gt_list[i]["dataset_name"])
#             # dataset name repeated based on the number of images in this dataset
#             dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
#             im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
#             im_path_list.extend(name_im_gt_list[i]["im_path"])
#             gt_path_list.extend(name_im_gt_list[i]["gt_path"])
#             im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
#             gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


#         self.dataset["data_name"] = dt_name_list
#         self.dataset["im_name"] = im_name_list
#         self.dataset["im_path"] = im_path_list
#         self.dataset["ori_im_path"] = deepcopy(im_path_list)
#         self.dataset["gt_path"] = gt_path_list
#         self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
#         self.dataset["im_ext"] = im_ext_list
#         self.dataset["gt_ext"] = gt_ext_list

#         self.eval_ori_resolution = eval_ori_resolution

#     def __len__(self):
#         return len(self.dataset["im_path"])
#     def __getitem__(self, idx):
#         im_path = self.dataset["im_path"][idx]
#         gt_path = self.dataset["gt_path"][idx]
#         im = io.imread(im_path)
#         gt = io.imread(gt_path)

#         if len(gt.shape) > 2:
#             gt = gt[:, :, 0]
#         if len(im.shape) < 3:
#             im = im[:, :, np.newaxis]
#         if im.shape[2] == 1:
#             im = np.repeat(im, 3, axis=2)
#         im = torch.tensor(im.copy(), dtype=torch.float32)
#         im = torch.transpose(torch.transpose(im,1,2),0,1)
#         gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)

#         sample = {
#         "imidx": torch.from_numpy(np.array(idx)),
#         "image": im,
#         "label": gt,
#         "shape": torch.tensor(im.shape[-2:]),
#         }
        
#         if self.transform:
#             sample = self.transform(sample)

#         if self.eval_ori_resolution:
#             sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
#             sample['ori_im_path'] = self.dataset["im_path"][idx]
#             sample['ori_gt_path'] = self.dataset["gt_path"][idx]

#         return sample


# # Copyright by HQ-SAM team
# # All rights reserved.

# ## data loader
# from __future__ import print_function, division

# import numpy as np
# import random
# from copy import deepcopy
# from skimage import io
# import os
# from glob import glob

# import torch
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torchvision import transforms, utils
# from torchvision.transforms.functional import normalize
# import torch.nn.functional as F
# from torch.utils.data.distributed import DistributedSampler

# #### --------------------- dataloader online ---------------------####

# def get_im_gt_name_dict(datasets, flag='valid'):
#     print("------------------------------", flag, "--------------------------------")
#     name_im_gt_list = []

#     for i in range(len(datasets)):
#         print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
#         tmp_im_list, tmp_gt_list = [], []
#         tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
#         print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

#         if(datasets[i]["gt_dir"]==""):
#             print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
#             tmp_gt_list = []
#         else:
#             tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
#             print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


#         name_im_gt_list.append({"dataset_name":datasets[i]["name"],
#                                 "im_path":tmp_im_list,
#                                 "gt_path":tmp_gt_list,
#                                 "im_ext":datasets[i]["im_ext"],
#                                 "gt_ext":datasets[i]["gt_ext"]})

#     return name_im_gt_list

# def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
#     gos_dataloaders = []
#     gos_datasets = []

#     if(len(name_im_gt_list)==0):
#         return gos_dataloaders, gos_datasets

#     num_workers_ = 1
#     if(batch_size>1):
#         num_workers_ = 2
#     if(batch_size>4):
#         num_workers_ = 4
#     if(batch_size>8):
#         num_workers_ = 8


#     if training:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
#             gos_datasets.append(gos_dataset)

#         gos_dataset = ConcatDataset(gos_datasets)
#         sampler = DistributedSampler(gos_dataset)
#         batch_sampler_train = torch.utils.data.BatchSampler(
#             sampler, batch_size, drop_last=True)
#         dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

#         gos_dataloaders = dataloader
#         gos_datasets = gos_dataset

#     else:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
#             sampler = DistributedSampler(gos_dataset, shuffle=False)
#             dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

#             gos_dataloaders.append(dataloader)
#             gos_datasets.append(gos_dataset)

#     return gos_dataloaders, gos_datasets

# class RandomHFlip(object):
#     def __init__(self,prob=0.5):
#         self.prob = prob
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         # random horizontal flip
#         if random.random() >= self.prob:
#             image = torch.flip(image,dims=[2])
#             label = torch.flip(label,dims=[2])

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

# class Resize(object):
#     def __init__(self,size=[320,320]):
#         self.size = size
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
#         label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}

# class RandomCrop(object):
#     def __init__(self,size=[288,288]):
#         self.size = size
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         h, w = image.shape[1:]
#         new_h, new_w = self.size

#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         image = image[:,top:top+new_h,left:left+new_w]
#         label = label[:,top:top+new_h,left:left+new_w]

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


# class Normalize(object):
#     def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
#         self.mean = mean
#         self.std = std

#     def __call__(self,sample):

#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
#         image = normalize(image,self.mean,self.std)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



# class LargeScaleJitter(object):
#     """
#         implementation of large scale jitter from copy_paste
#         https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
#     """

#     def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
#         self.desired_size = torch.tensor(output_size)
#         self.aug_scale_min = aug_scale_min
#         self.aug_scale_max = aug_scale_max

#     def pad_target(self, padding, target):
#         target = target.copy()
#         if "masks" in target:
#             target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
#         return target

#     def __call__(self, sample):
#         imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         #resize keep ratio
#         out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

#         random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
#         scaled_size = (random_scale * self.desired_size).round()

#         scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
#         scaled_size = (image_size * scale).round().long()
        
#         scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
#         scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
#         # random crop
#         crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

#         margin_h = max(scaled_size[0] - crop_size[0], 0).item()
#         margin_w = max(scaled_size[1] - crop_size[1], 0).item()
#         offset_h = np.random.randint(0, margin_h + 1)
#         offset_w = np.random.randint(0, margin_w + 1)
#         crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
#         crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

#         scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
#         scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

#         # pad
#         padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
#         padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
#         image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
#         label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}






# class OnlineDataset(Dataset):
#     def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

#         self.transform = transform
#         self.dataset = {}
#         ## combine different datasets into one
#         dataset_names = []
#         dt_name_list = [] # dataset name per image
#         im_name_list = [] # image name
#         im_path_list = [] # im path
#         gt_path_list = [] # gt path
#         im_ext_list = [] # im ext
#         gt_ext_list = [] # gt ext
#         for i in range(0,len(name_im_gt_list)):
#             dataset_names.append(name_im_gt_list[i]["dataset_name"])
#             # dataset name repeated based on the number of images in this dataset
#             dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
#             im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
#             im_path_list.extend(name_im_gt_list[i]["im_path"])
#             gt_path_list.extend(name_im_gt_list[i]["gt_path"])
#             im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
#             gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


#         self.dataset["data_name"] = dt_name_list
#         self.dataset["im_name"] = im_name_list
#         self.dataset["im_path"] = im_path_list
#         self.dataset["ori_im_path"] = deepcopy(im_path_list)
#         self.dataset["gt_path"] = gt_path_list
#         self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
#         self.dataset["im_ext"] = im_ext_list
#         self.dataset["gt_ext"] = gt_ext_list

#         self.eval_ori_resolution = eval_ori_resolution

#     def __len__(self):
#         return len(self.dataset["im_path"])
#     def __getitem__(self, idx):
#         im_path = self.dataset["im_path"][idx]
#         gt_path = self.dataset["gt_path"][idx]
#         im = io.imread(im_path)
#         gt = io.imread(gt_path)

        

#         if len(gt.shape) > 2:
#             gt = gt[:, :, 0]
#         if len(im.shape) < 3:
#             im = im[:, :, np.newaxis]
#         if im.shape[2] == 1:
#             im = np.repeat(im, 3, axis=2)
#         im = torch.tensor(im.copy(), dtype=torch.float32)
#         im = torch.transpose(torch.transpose(im,1,2),0,1)
#         gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)

#         sample = {
#         "imidx": torch.from_numpy(np.array(idx)),
#         "image": im,
#         "label": gt,
#         "shape": torch.tensor(im.shape[-2:]),
#         }
        
#         if self.transform:
#             sample = self.transform(sample)

#         if self.eval_ori_resolution:
#             sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
#             sample['ori_im_path'] = self.dataset["im_path"][idx]
#             sample['ori_gt_path'] = self.dataset["gt_path"][idx]

#         return sample



# # Copyright by HQ-SAM team
# # All rights reserved.

# ## data loader_ch4
# from __future__ import print_function, division

# import numpy as np
# import random
# from copy import deepcopy
# from skimage import io
# import os
# from glob import glob

# import torch
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torchvision import transforms, utils
# from torchvision.transforms.functional import normalize
# import torch.nn.functional as F
# from torch.utils.data.distributed import DistributedSampler

# #### --------------------- dataloader online ---------------------####

# def get_im_gt_name_dict(datasets, flag='valid'):
#     print("------------------------------", flag, "--------------------------------")
#     name_im_gt_list = []

#     for i in range(len(datasets)):
#         print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
#         tmp_im_list, tmp_gt_list = [], []
#         tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
#         print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
#         tmp_im_ch4_list, tmp_im_ch5_list = [],[]
#         tmp_im_ch4_list = glob(datasets[i]["im_ch4_dir"]+os.sep+'*'+datasets[i]["im_ext"])
#         print('-im ch4-',datasets[i]["name"],datasets[i]["im_ch4_dir"], ': ',len(tmp_im_ch4_list))
        
#         tmp_im_ch5_list = glob(datasets[i]["im_ch5_dir"]+os.sep+'*'+datasets[i]["im_ext"])
#         print('-im ch5-',datasets[i]["name"],datasets[i]["im_ch5_dir"], ': ',len(tmp_im_ch5_list))

#         if(datasets[i]["gt_dir"]==""):
#             print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
#             tmp_gt_list = []
#         else:
#             tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
#             print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


#         name_im_gt_list.append({"dataset_name":datasets[i]["name"],
#                                 "im_path":tmp_im_list,
#                                 "gt_path":tmp_gt_list,
#                                 "im_ext":datasets[i]["im_ext"],
#                                 "gt_ext":datasets[i]["gt_ext"],
#                                 "im_ch4_path":tmp_im_ch4_list,
#                                 "im_ch5_path":tmp_im_ch5_list
#         })
                               

#     return name_im_gt_list

# def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
#     gos_dataloaders = []
#     gos_datasets = []

#     if(len(name_im_gt_list)==0):
#         return gos_dataloaders, gos_datasets

#     num_workers_ = 1
#     if(batch_size>1):
#         num_workers_ = 2
#     if(batch_size>4):
#         num_workers_ = 4
#     if(batch_size>8):
#         num_workers_ = 8


#     if training:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
#             gos_datasets.append(gos_dataset)

#         gos_dataset = ConcatDataset(gos_datasets)
#         sampler = DistributedSampler(gos_dataset)
#         batch_sampler_train = torch.utils.data.BatchSampler(
#             sampler, batch_size, drop_last=True)
#         dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

#         gos_dataloaders = dataloader
#         gos_datasets = gos_dataset

#     else:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
#             sampler = DistributedSampler(gos_dataset, shuffle=False)
#             dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

#             gos_dataloaders.append(dataloader)
#             gos_datasets.append(gos_dataset)

#     return gos_dataloaders, gos_datasets

# class RandomHFlip(object):
#     def __init__(self,prob=0.5):
#         self.prob = prob
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
#         im_5channel= sample['image_5channel']

#         # random horizontal flip
#         if random.random() >= self.prob:
#             image = torch.flip(image,dims=[2])
#             label = torch.flip(label,dims=[2])
#             im_5channel= torch.flip(im_5channel,dims=[2])

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape, 'image_5channel':im_5channel}

# class Resize(object):
#     def __init__(self,size=[320,320]):
#         self.size = size
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
#         im_5channel= sample['image_5channel']

#         image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
#         label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)
#         im_5channel = torch.squeeze(F.interpolate(torch.unsqueeze(im_5channel,0),self.size,mode='bilinear'),dim=0)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size), 'image_5channel':im_5channel}

# class RandomCrop(object):
#     def __init__(self,size=[288,288]):
#         self.size = size
#     def __call__(self,sample):
#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#         h, w = image.shape[1:]
#         new_h, new_w = self.size

#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         image = image[:,top:top+new_h,left:left+new_w]
#         label = label[:,top:top+new_h,left:left+new_w]

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


# class Normalize(object):
#     def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
#         self.mean = mean
#         self.std = std

#     def __call__(self,sample):

#         imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
#         image = normalize(image,self.mean,self.std)

#         return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



# class LargeScaleJitter(object):
#     """
#         implementation of large scale jitter from copy_paste
#         https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
#     """

#     def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
#         self.desired_size = torch.tensor(output_size)
#         self.aug_scale_min = aug_scale_min
#         self.aug_scale_max = aug_scale_max

#     def pad_target(self, padding, target):
#         target = target.copy()
#         if "masks" in target:
#             target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
#         return target

#     def __call__(self, sample):
#         imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']
#         im_5channel= sample['image_5channel']

#         #resize keep ratio
#         out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

#         random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
#         scaled_size = (random_scale * self.desired_size).round()

#         scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
#         scaled_size = (image_size * scale).round().long()
        
#         scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
#         scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
#         scaled_im_5channel = torch.squeeze(F.interpolate(torch.unsqueeze(im_5channel,0),scaled_size.tolist(),mode='bilinear'),dim=0)
#         # random crop
#         crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

#         margin_h = max(scaled_size[0] - crop_size[0], 0).item()
#         margin_w = max(scaled_size[1] - crop_size[1], 0).item()
#         offset_h = np.random.randint(0, margin_h + 1)
#         offset_w = np.random.randint(0, margin_w + 1)
#         crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
#         crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

#         scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
#         scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]
#         scaled_im_5channel = scaled_im_5channel[:,crop_y1:crop_y2, crop_x1:crop_x2]

#         # pad
#         padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
#         padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
#         image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
#         label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

#         im_5channel = F.pad(scaled_im_5channel, [0,padding_w, 0,padding_h],value=128)
        
#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:]), 'image_5channel':im_5channel}






# class OnlineDataset(Dataset):
#     def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

#         self.transform = transform
#         self.dataset = {}
#         ## combine different datasets into one
#         dataset_names = []
#         dt_name_list = [] # dataset name per image
#         im_name_list = [] # image name
#         im_path_list = [] # im path
#         gt_path_list = [] # gt path
#         im_ext_list = [] # im ext
#         gt_ext_list = [] # gt ext
#         im_ch4_name_list = []
#         im_ch5_name_list = []
#         im_ch4_path_list = []
#         im_ch5_path_list = []
        
#         for i in range(0,len(name_im_gt_list)):
#             dataset_names.append(name_im_gt_list[i]["dataset_name"])
#             # dataset name repeated based on the number of images in this dataset
#             dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
#             im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
#             im_path_list.extend(name_im_gt_list[i]["im_path"])
#             gt_path_list.extend(name_im_gt_list[i]["gt_path"])
#             im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
#             gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])
            
#             im_ch4_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_ch4_path"]])
#             im_ch5_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_ch5_path"]])
#             im_ch4_path_list.extend(name_im_gt_list[i]["im_ch4_path"])
#             im_ch5_path_list.extend(name_im_gt_list[i]["im_ch5_path"])


#         self.dataset["data_name"] = dt_name_list
#         self.dataset["im_name"] = im_name_list
#         self.dataset["im_path"] = im_path_list
#         self.dataset["ori_im_path"] = deepcopy(im_path_list)
#         self.dataset["gt_path"] = gt_path_list
#         self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
#         self.dataset["im_ext"] = im_ext_list
#         self.dataset["gt_ext"] = gt_ext_list
        
#         self.dataset["im_ch4_name"] = im_ch4_name_list
#         self.dataset["im_ch5_name"] = im_ch5_name_list
        
#         self.dataset["im_ch4_path"] = im_ch4_path_list
#         self.dataset["im_ch5_path"] = im_ch5_path_list

#         self.eval_ori_resolution = eval_ori_resolution

#     def __len__(self):
#         return len(self.dataset["im_path"])
#     def __getitem__(self, idx):
#         im_path = self.dataset["im_path"][idx]
#         gt_path = self.dataset["gt_path"][idx]
#         im = io.imread(im_path)
#         gt = io.imread(gt_path)
#         # original code 10/01
#         im_ch4_path = self.dataset["im_ch4_path"][idx]
#         im_ch4 = io.imread(im_ch4_path)
        
#         im_ch5_path = self.dataset["im_ch5_path"][idx]
#         im_ch5 = io.imread(im_ch5_path)
        

#         if len(gt.shape) > 2:
#             gt = gt[:, :, 0]
#         if len(im.shape) < 3:
#             im = im[:, :, np.newaxis]
#         if im.shape[2] == 1:
#             im = np.repeat(im, 3, axis=2)
        
#         if im_ch4.shape[2] > 1:
#             im_ch4 = im_ch4[:,:,:1]  
            
#         if im_ch5.shape[2] > 1:
#             im_ch5 = im_ch5[:,:,:1]  
#         #print(f'image shape: {im.shape}; ch4 shape: {im_ch4.shape}; ch5 shape: {im_ch5.shape} #################################################3')

#         # Create two additional grayscale channels, TODO: change to images
#         # height, width = im.shape[:2]
#         # gray1 = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)
#         # gray2 = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)

#         # Concatenate the original image with the two new grayscale channels
#         #im_5channel = np.concatenate((im, gray1, gray2), axis=2)
#         im_5channel = np.concatenate((im, im_ch4, im_ch5), axis=2)
#         # Convert to PyTorch tensor and rearrange dimensions
#         im_5channel = torch.tensor(im_5channel.copy(), dtype=torch.float32)
#         im_5channel = torch.permute(im_5channel, (2, 0, 1))

#         im = torch.tensor(im.copy(), dtype=torch.float32)
#         im = torch.transpose(torch.transpose(im,1,2),0,1)

#         gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)
        
#         #print(f'im_5channel shape: {im_5channel.shape} aaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
#         im_5channel = im_5channel[:4,:,:]    
#         #print(f'im_5channel shape: {im_5channel.shape} bbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        
#         sample = {
#         "imidx": torch.from_numpy(np.array(idx)),
#         "image": im,
#         "label": gt,
#         "shape": torch.tensor(im.shape[-2:]),
#         "image_5channel": im_5channel
#         }
        
#         if self.transform:
#             sample = self.transform(sample)

#         if self.eval_ori_resolution:
#             sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
#             sample['ori_im_path'] = self.dataset["im_path"][idx]
#             sample['ori_gt_path'] = self.dataset["gt_path"][idx]

#         return sample




# Copyright by HQ-SAM team
# All rights reserved.

## data loader_red-glcm
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from skimage.transform import resize as sk_resize  # 文件顶部确保有这行




#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
        tmp_im_ch4_list = []
        tmp_im_ch4_list = glob(datasets[i]["im_ch4_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im ch4-',datasets[i]["name"],datasets[i]["im_ch4_dir"], ': ',len(tmp_im_ch4_list))
        
        # tmp_im_ch5_list = glob(datasets[i]["im_ch5_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        # print('-im ch5-',datasets[i]["name"],datasets[i]["im_ch5_dir"], ': ',len(tmp_im_ch5_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"],
                                "im_ch4_path":tmp_im_ch4_list,
                                # "im_ch5_path":tmp_im_ch5_list
        })
                               

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class Resize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}

class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}









# Modified OnlineDataset class to replace the Red channel with the GLCM channel
class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, replace_red_with_glcm=True):
        self.transform = transform
        # self.replace_red_with_glcm = replace_red_with_glcm  # Control whether to replace Red channel
        self.replace_red_with_glcm = bool(replace_red_with_glcm)
        self.dataset = {}
        
        # Combine different datasets into one
        dataset_names = []
        dt_name_list = []  # dataset name per image
        im_name_list = []  # image name
        im_path_list = []  # image path
        gt_path_list = []  # ground truth path
        im_ext_list = []  # image extension
        gt_ext_list = []  # ground truth extension
        im_ch4_list = []  # GLCM path

        for i in range(len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])
            im_ch4_list.extend(name_im_gt_list[i]["im_ch4_path"])  # Include GLCM paths

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list
        self.dataset["im_ch4_path"] = im_ch4_list  # Store GLCM paths

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    # def __getitem__(self, idx):
    #     im_path = self.dataset["im_path"][idx]
    #     gt_path = self.dataset["gt_path"][idx]
    #     if len(self.dataset["im_ch4_path"]) > idx:
    #         im_ch4_path = self.dataset["im_ch4_path"][idx]
    #     else:
    #         im_ch4_path = None  # Get GLCM path
        
    #     im = io.imread(im_path)
    #     gt = io.imread(gt_path)
    #     # 取路径（安全获取）
    #     im_ch4_path = None
    #     if ("im_ch4_path" in self.dataset) and (idx < len(self.dataset["im_ch4_path"])):
    #         im_ch4_path = self.dataset["im_ch4_path"][idx]

    #     # 是否真的需要 ch4：仅当开启 GLCM 替换红通道时才读取
    #     use_ch4 = getattr(self, "replace_red_with_glcm", False)

    #     im_ch4 = None
    #     if use_ch4 and isinstance(im_ch4_path, str) and len(im_ch4_path) > 0:
    #         # 如果你希望 ch4 与 im 不同，这里会读入；如果 im_ch4_path == im_path，也能正常读
    #         im_ch4 = io.imread(im_ch4_path)
    #     else:
    #         im_ch4 = None  # 显式地没有 ch4

    #     # —— 新增：尺寸 & 类型对齐 —— #
    #     H, W = im.shape[:2]
    #     h4, w4 = im_ch4.shape[:2]
    #     if (h4 != H) or (w4 != W):
    #         # 用最近邻，不产生小数，避免标签类纹理被平滑
    #         im_ch4 = sk_resize(
    #             im_ch4, (H, W), order=0, preserve_range=True, anti_aliasing=False
    #         ).astype(im.dtype)
    #     else:
    #         # 保证 dtype 与 RGB 一致（通常 uint8）
    #         if im_ch4.dtype != im.dtype:
    #             im_ch4 = im_ch4.astype(im.dtype)

    #     # Optionally replace Red channel with GLCM channel
    #     if self.replace_red_with_glcm:
    #         if len(im.shape) == 3 and im.shape[2] == 3:
    #             im[:, :, 0] = im_ch4  # Replace the Red channel with GLCM
    #         else:
    #             raise ValueError("Expected a 3-channel RGB image")
        

    #     if len(gt.shape) > 2:
    #         gt = gt[:, :, 0]
    #     if len(im.shape) < 3:
    #         im = im[:, :, np.newaxis]
    #     if im.shape[2] == 1:
    #         im = np.repeat(im, 3, axis=2)
        
    #     im = torch.tensor(im.copy(), dtype=torch.float32)
    #     im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)
    #     gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)

    #     sample = {
    #         "imidx": torch.from_numpy(np.array(idx)),
    #         "image": im,
    #         "label": gt,
    #         "shape": torch.tensor(im.shape[-2:]),
    #     }

    #     if self.transform:
    #         sample = self.transform(sample)

    #     if self.eval_ori_resolution:
    #         sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. No flip here
    #         sample['ori_im_path'] = self.dataset["im_path"][idx]
    #         sample['ori_gt_path'] = self.dataset["gt_path"][idx]

    #     return sample

    def __getitem__(self, idx):
        # ---- 基础路径 ----
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]

        # 安全获取 ch4 路径（可能不存在）
        im_ch4_path = None
        if ("im_ch4_path" in self.dataset) and (idx < len(self.dataset["im_ch4_path"])):
            im_ch4_path = self.dataset["im_ch4_path"][idx]

        # ---- 读图/标注 ----
        im = io.imread(im_path)
        gt = io.imread(gt_path)

        # 训练时是否打算用 ch4 替换红通道（由 Dataset 初始化时的开关控制）
        use_ch4_cfg = bool(getattr(self, "replace_red_with_glcm", False))

        # 只有在开关打开且路径是有效字符串时才去读 ch4
        im_ch4 = None
        if use_ch4_cfg and isinstance(im_ch4_path, str) and len(im_ch4_path) > 0:
            try:
                im_ch4 = io.imread(im_ch4_path)
            except Exception as e:
                # 路径坏了/读失败，降级为不用 ch4，避免崩
                im_ch4 = None

        # ---- 尺寸/类型对齐（仅在 im_ch4 确实存在时才做）----
        if im_ch4 is not None:
            # 如果 ch4 是 3 通道，取单通道（一般 ch4 应该是单通道）
            if im_ch4.ndim == 3 and im_ch4.shape[2] > 1:
                im_ch4 = im_ch4[:, :, 0]

            H, W = im.shape[:2]
            h4, w4 = im_ch4.shape[:2]
            if (h4 != H) or (w4 != W):
                # 最近邻以避免纹理被平滑
                im_ch4 = sk_resize(
                    im_ch4, (H, W), order=0, preserve_range=True, anti_aliasing=False
                )
            # 保证 dtype 与 RGB 一致（通常 uint8）
            if im_ch4.dtype != im.dtype:
                im_ch4 = im_ch4.astype(im.dtype)

        # ---- 可选：用 ch4 替换红通道 ----
        # 只有当：1) 你开启了 replace_red_with_glcm；2) 成功读到了 ch4，才会替换
        if use_ch4_cfg and (im_ch4 is not None):
            # 如果原图是灰度，先扩到 3 通道
            if im.ndim == 2:
                im = im[:, :, np.newaxis]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            # 期望 3 通道 RGB
            if im.ndim == 3 and im.shape[2] == 3:
                im[:, :, 0] = im_ch4  # 替换 R 通道
            else:
                raise ValueError("Expected a 3-channel RGB image after expansion")
        else:
            # 没有 ch4 时，保证图像至少是 3 通道
            if im.ndim == 2:
                im = im[:, :, np.newaxis]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)

        # ---- 处理 gt 到单通道 ----
        if gt.ndim > 2:
            gt = gt[:, :, 0]

        # ---- 转 tensor（C,H,W）----
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = im.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)  # (1,H,W)

        sample = {
            "imidx": torch.tensor(idx),
            "image": im,
            "label": gt,
            "shape": torch.tensor(im.shape[-2:]),  # (H,W)
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)
            sample["ori_im_path"] = im_path
            sample["ori_gt_path"] = gt_path

        return sample

