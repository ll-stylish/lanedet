#!/usr/bin/env python3
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
import os
import sys

import time
import matplotlib.pyplot as plt
import ast

import rospy
from geometry_msgs.msg import Twist




sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config


class UFLDv2:
    def __init__(self, engine_path, config_path, ori_size):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        cfg = Config.fromfile(config_path)
        self.ori_img_w, self.ori_img_h = ori_size
        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col
        self.row_anchor = np.linspace(0.42, 1, self.num_row)
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def pred2coords(self, pred):
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1)
        # n , num_cls, num_lanes
        valid_row = pred['exist_row'].argmax(1)
        # n, num_cls, num_lanes

        max_indices_col = pred['loc_col'].argmax(1)
        # n , num_cls, num_lanes
        valid_col = pred['exist_col'].argmax(1)
        # n, num_cls, num_lanes

        pred['loc_row'] = pred['loc_row']
        pred['loc_col'] = pred['loc_col']

        coords = []
        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]

        for i in row_lane_idx:
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - self.input_width),
                                                          min(num_grid_row - 1,
                                                              max_indices_row[0, k, i] + self.input_width) + 1)))

                        out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        tmp.append((int(out_tmp), int(self.row_anchor[k] * self.ori_img_h)))
                coords.append(tmp)

        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - self.input_width),
                                                          min(num_grid_col - 1,
                                                              max_indices_col[0, k, i] + self.input_width) + 1)))
                        out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * self.ori_img_h
                        tmp.append((int(self.col_anchor[k] * self.ori_img_w), int(out_tmp)))
                coords.append(tmp)
        return coords

    def forward(self, img):
        im0 = img.copy()
        img = img[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 2, 0, 1))
        img = np.ascontiguousarray(img)
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        preds = {}
        for out in self.outputs:
            output = np.zeros(out['shape'], out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            preds[out['name']] = torch.tensor(output)
        coords = self.pred2coords(preds)
        # print(coords)
        for lane in coords:
            for coord in lane:
                cv2.circle(im0, coord, 2, (0, 255, 0), -1)
        cv2.imshow("result", im0)
        return coords


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/culane_res34.py', help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weight/culane_res34.engine',
                        help='path to engine file', type=str)
    parser.add_argument('--video_path', default='output1.mp4', help='path to video file', type=str)
    parser.add_argument('--ori_size', default=(1600, 320), help='size of original frame', type=tuple)
    return parser.parse_args()


class PID_Controler:
    def __init__(self):
        self.Kp = 0.01
        self.Ki = 0
        self.Kd = 0
        self.error = 0
        self.last_error = 0
        self.integral_val = 0
        self.delta_val = 0
        self.input = 0
        self.output = 0
        self.last_output=0

    def control(self, error_val):
        self.error = error_val
        self.integral_val += self.error
        self.delta_val = self.error - self.last_error

        self.output = self.Kp * self.error + self.Ki * self.integral_val + self.Kd * self.delta_val
        self.last_error = self.error

        self.last_output=self.output

#一阶段滤波器
def first_order_filter(data,last_data,alpha=0.25):
    output = alpha * data + (1-alpha) * last_data
    return output

def Lane_fitting(coords,weight):
    # 获取车道线坐标
    data_line = []  # 每条线的数据
    for i in range(len(coords)):
        temp = np.zeros([len(coords[i]), 2]).astype(int)
        # print(temp)
        for index, value in enumerate(coords[i]):
            temp[index][0], temp[index][1] = value[0], value[1]
        data_line.append(temp)  # 获取车道线在图像上的坐标 图片坐标系


    # 车道线拟合
    line_param = []  # 每条线的参数列表
    bottom_point = []
    for index in range(len(data_line)):
        temp_param = np.polyfit(data_line[index][:, 1],
                                data_line[index][:, 0], 2)  # 输入纵坐标y 返回横坐标x
        line_param.append(temp_param)  # 车道线拟合后的参数
        bottom_point.append(
            int(np.polyval(temp_param, img_high))
        )  # 得到最低点的坐标值

    # 左边线
    bottom_left_points = [left for left in bottom_point if left <= mid_width]
    bottom_left_point = max(bottom_left_points)
    left_point_index = [i for i, val in enumerate(bottom_point) if val == bottom_left_point][0]
    left_line = data_line[left_point_index]  # 数据
    left_line_param = line_param[left_point_index]
    # for x, y in left_line:
    #     cv2.circle(img, (x, y), 1, (0, 125, 0), 4)

    # 右边线
    bottom_right_points = [right for right in bottom_point if right > mid_width]
    bottom_right_point = max(bottom_right_points)
    right_point_index = [i for i, val in enumerate(bottom_point) if val == bottom_right_point][0]
    right_line = data_line[right_point_index]  # 数据
    right_line_param = line_param[right_point_index]
    # for x, y in right_line:
    #     cv2.circle(img, (x, y), 11, (0, 125, 0), 4)


    
    # 图像偏差有效值区域
    control_range_min = int(0.7 * img_high)
    control_range_max = int(0.9 * img_high)
    weight_matrix = np.empty([control_range_max - control_range_min + 1])#每行的权重矩阵

    # 拟合数据左边线
    poly_func = np.poly1d(left_line_param)
    fit_x = np.arange(control_range_min, control_range_max)  # 有效控制区域
    fit_x = np.concatenate(fit_x, control_range_max)
    left_fit_y = poly_func(fit_x)  # 计算拟合曲线上的 x 值 （横）
    # 拟合数据右边线
    poly_func = np.poly1d(right_line_param)
    right_fit_y = poly_func(fit_x)  # 计算拟合曲线上的 x 值 （横）

    middle_line = np.empty([len(fit_x), 3])  # x y error
    middle_line[:, 0] = (left_fit_y + right_fit_y) / 2
    middle_line[:, 1] = fit_x
    middle_line[:, 2] = mid_width - middle_line[:, 0]  # 中线误差
    # plt.plot(range(len(middle_line[:,2])),middle_line[:,2])
    # plt.show()

    # 绘制拟合的 左 右边线 和 中线
    # for j in range(len(fit_x)):
    #     cv2.circle(img, (left_fit_y.astype(int)[j], fit_x.astype(int)[j]), 2, (255, 0, 255), 2)  # left_line
    #     cv2.circle(img, (right_fit_y.astype(int)[j], fit_x.astype(int)[j]), 2, (125, 0, 255), 2)  # right_line
    #     cv2.circle(img, (int(middle_line[j, 0]), int(middle_line[j, 1])), 2, (25, 0, 255), 2)  # middle_line

    #     cv2.circle(img, (int(mid_width), int(middle_line[j, 1])), 2, (0, 125, 255), 2)
    
    error_val = np.sum(
        np.multiply((middle_line[:, 0] - mid_width), (weight_matrix / np.sum(weight_matrix)))
        )  # 归一化误差
    
    return error_val


if __name__ == "__main__":
    rospy.init_node('controler_node',anonymous=True)
    # rate=rospy.Rate(10)#10hz
    controler_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
    Controler=PID_Controler()

    vel_msg=Twist()
    vel_msg.linear.x = 0
    vel_msg.angular.z= 0
    controler_pub.publish(vel_msg)
    rospy.loginfo("cmd_vel发布初始化")

    #图像尺寸获取
    img_high=cv2.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    img_width=cv2.get(cv2.CAP_PROP_FRAME_WIDTH)
    mid_width=img_high/2

    #终端参数读取
    args = get_args()

    # cap = cv2.VideoCapture(args.video_path)
    # out = cv2.VideoWriter('./output2 .mp4', cv2.VideoWriter_fourcc(*"mp4v"),30, (1600, 903))
    cap = cv2.VideoCapture(2)
    # cap.set(3,1920)
    # cap.set(4,1080)

    isnet = UFLDv2(args.engine_path, args.config_path, args.ori_size)

    
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1600, 903))
        img = img[380:700, :, :]
        coords=isnet.forward(img)# 获取车道点
        
        # 区域权重
        weight = [0.1, 0.2, 0.4, 0.2, 0.1]  # 偏差权重 sum=1

        error_val=Lane_fitting(coords)

        Controler.control(error_val)  # 输出值

        vel_msg.linear.x=1.0
        vel_msg.angular.z = 0.0#Controler.output
        controler_pub.publish(vel_msg)
    # out.release() 
