#!/usr/bin/env python3

import torch, os, cv2,time
from utils.dist_utils import dist_print
import torch, os
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset


import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np


"""torch 获取车道线信息"""
def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1280, original_image_height = 720):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes
    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)
    return coords

"""处理图片数据"""
def img_callback(data):
    #图像转化为cv->图像输入torch-》接入函数-》输出图像

    # assert isinstance(data,Image)
    cv_img=np.frombuffer(data.data,dtype=np.uint8).reshape(data.height,data.width,-1)
    cv2.imwrite(os.path.join(cfg.data_root)+'/1.jpg',cv_img)#保存摄影机图像
    # 车道线识别
    for split, dataset in zip(splits, datasets):

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)  # 数据输入torch

        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # print(split[:-3]+'avi')
        # vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)
            
            # vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            # cv2.imshow('vis',vis)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break

            
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
            

            # for lane in coords:
            #     for coord in lane:
            #         cv2.circle(vis,coord,5,(0,255,0),-1)
            # cv2.imwrite((cfg.data_root)+'/1.jpg',vis)
            # cv2.imshow('vis',vis)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break

    
            # vout.write(vis)
        # vout.release()
    # time_end=time.time()
    # print(time_mid-time_origin,'|',time_end-time_mid)


"""摄像头图像订阅"""
def img_sub():
    rospy.init_node('img_subscribe_node',anonymous=True)#节点
    img_subscriber=rospy.Subscriber('/image_view/image_raw',Image,img_callback)#topic数据
    rospy.spin()

"""车道线检测初始化"""
def lane_detect_init():
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start tesing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
    
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError
    
    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((int(320 / 0.6), 1600)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt']
        datasets = [LaneTestDataset('database','database/list/test_split/test0_normal.txt',img_transform = img_transforms, crop_size = 320)]
        # img_w, img_h = 640, 360#设置图像的尺寸

    # elif cfg.dataset == 'Tusimple':
    #     splits = ['test.txt']
    #     datasets = [LaneTestDataset(cfg.data_root,
    #                                 os.path.join(cfg.data_root, split),
    #                                 img_transform = img_transforms, 
    #                                 crop_size = cfg.train_height) 
    #                                 for split in splits]
    #     img_w, img_h = 1280, 720
    else:
        raise NotImplementedError

    dist_print("车道线检测初始化完成！")
    
    return splits,datasets,net,cfg
  


if __name__ == "__main__":
    time_origin=time.time()
    bridge=CvBridge()#cv桥
    
    splits,datasets,net,cfg=lane_detect_init()#torch初始化
    # for a,b in zip(splits, datasets):
    #     dist_print('datapath:',a,'|',b)
    img_w,img_h=640,360#图像格式设置


    img_sub()