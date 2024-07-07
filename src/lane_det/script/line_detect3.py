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
        coords = self.pred2coords(preds)#获取预测点 形式[[ () ()  ]]

        # print(coords)
        # if coords:
        #     e = Lane_fitting(coords)
        #     print('归一化误差：',e)
        # else:
        #     pass
        
        # for lane in coords:
        #     for coord in lane:
        #         cv2.circle(im0, coord, 2, (0, 255, 0), -1)
        # for i in range(200,300):
        #     cv2.circle(im0,(800,i) , 2, (0, 5, 120), -1)
        # cv2.imshow("result", im0)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     pass

        return coords



def get_args():
    parser = argparse.ArgumentParser()#获取参合
    parser.add_argument('--config_path', default='configs/culane_res34.py', help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weight/culane_res34.engine',help='path to engine file', type=str)
    parser.add_argument('--video_path', default='example.mp4', help='path to video file', type=str)
    parser.add_argument('--ori_size', default=(1600, 320), help='size of original frame', type=tuple)
    return parser.parse_args()


class PID_Controler:
    def __init__(self):
        #pid参数初始化
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

    def output_control(self, error_val):
        """控制转向输出

        Args:
            error_val (数值): 当前车辆朝向与中心线的 像素差
        """
        self.error = error_val
        self.integral_val += self.error#当前误差
        self.delta_val = self.error - self.last_error#误差增量

        self.output = self.Kp * self.error + self.Ki * self.integral_val + self.Kd * self.delta_val
        
        self.last_output=self.output#更新 历史输出
        self.last_error = self.error#更新 历史误差
        return self.output

        

    def velocity_pub(self,velocity_publisher,vel_msg,forward_speed_x,rotate_speed_z):
        """输出cmd_vel控制小车 前进+转向

        Args:
            velocity_publisher (rospy.Publisher): cmd_vel发布者
            vel_msg(Twist): 速度信息
            forward_speed_x (数值): 小车前进速度(m/s)
            rotate_speed_z (数值): 小车转向速度(rad/s)
        """
        vel_msg.linear.x = forward_speed_x
        vel_msg.angular.z = rotate_speed_z
        velocity_publisher.publish(vel_msg)



def first_order_filter(data,last_data,alpha=0.25):
    output = alpha * data + (1-alpha) * last_data
    return output

def Lane_fitting(coords,weight):
    """进行车道线预测

    Args:
        coords (isnet.forward返回的点的参数 格式：[[ (),()]]): 车道线点数据
        weight (list): 权重矩阵

    Returns:
        error_val: 转向误差值
    """

    data_line = []  # 扫描到的 每条线的数据

    for i in range(len(coords)):
        temp = np.zeros([len(coords[i]), 2]).astype(int)
        for index, value in enumerate(coords[i]):
            temp[index][0], temp[index][1] = value[0], value[1]

        # 获取车道线在图像上的坐标 （图片坐标系）
        data_line.append(temp)
    # print(data_line)
    # 车道线拟合
    line_param = []  # 每条线的参数列表
    bottom_point = []
    for index in range(len(data_line)):
        temp_param = np.polyfit(data_line[index][:, 1],
                                data_line[index][:, 0], 2)  # 输入纵坐标y 返回横坐标x
        line_param.append(temp_param)  # 车道线拟合后的参数

        rospy.loginfo('data_line[index][len(data_line[index])-1][0]:%d'%data_line[index][len(data_line[index])-1][0])#rospy.loginfo('data_line[index][len(data_line[index])-1][1]:%d'%data_line[index][len(data_line[index])-1][0])
        bottom_point.append(
            int(np.polyval(temp_param,320))#data_line[index][len(data_line[index])-1][0]
        )  # 得到最低点的坐标值
    rospy.loginfo('bottom_point:%s'%bottom_point)

    # 左边线
    bottom_left_points = [left for left in bottom_point if left <= 800]
    if bottom_left_points: #是否扫到线
        bottom_left_point = max(bottom_left_points)
        left_point_index = [i for i, val in enumerate(bottom_point) if val == bottom_left_point][0]
        left_line_param = line_param[left_point_index]#边线拟合参数
        # left_line = data_line[left_point_index]  # 左边线数据
        # for x, y in left_line:
        #     cv2.circle(img, (x, y), 1, (0, 125, 0), 4)
    else:
        #**********************************to do*******************************************
        #**********************************************************************************
        #没边线怎么办
        return None

    # 右边线
    bottom_right_points = [right for right in bottom_point if right > 800]
    if bottom_right_points: #是否扫到线
        bottom_right_point = max(bottom_right_points)
        right_point_index = [i for i, val in enumerate(bottom_point) if val == bottom_right_point][0]
        right_line_param = line_param[right_point_index]#边线拟合参数
        # right_line = data_line[right_point_index]  # 右边线数据
        # for x, y in right_line:
        #     cv2.circle(img, (x, y), 11, (0, 125, 0), 4)
    else:
        #**********************************to do*******************************************
        #**********************************************************************************
        #没边线怎么办
        return None
    
    # cv2.imshow("result", img)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     pass

    
    # 图像偏差有效值区域
    control_range_min = int(0.7 * 320)
    control_range_max = int(0.9 * 320)
    # control_range_min = int(0.7 * img_high)
    # control_range_max = int(0.9 * img_high)
    rospy.loginfo('control_range_max - control_range_min:%d'%(control_range_max - control_range_min))
    weight_matrix = np.empty([control_range_max - control_range_min ])#每行的权重矩阵 #empty
    step = int((control_range_max - control_range_min ) / len(weight))  # 元素长度
    for i in range(len(weight)):
        if i != len(weight) - 1:
            weight_matrix[(i * step):((i + 1) * step)] = weight[i]
        else:
            weight_matrix[(i * step):] = weight[i]

    # 拟合数据左边线
    poly_func = np.poly1d(left_line_param)
    fit_x = np.arange(control_range_min, control_range_max)  # 有效控制区域
    fit_x = np.concatenate(fit_x, control_range_max)
    left_fit_y = poly_func(fit_x)  # 计算拟合曲线上的 x 值 （横）
    # 拟合数据右边线
    poly_func = np.poly1d(right_line_param)
    right_fit_y = poly_func(fit_x)  # 计算拟合曲线上的 x 值 （横）


    middle_line = np.zeros([len(fit_x), 3])  # 车道中线数据 x y error->与图像中线距离三 
    # print(len(weight_matrix))
    middle_line[:, 0] = (left_fit_y + right_fit_y) / 2
    middle_line[:, 1] = fit_x
    middle_line[:, 2] = 800 - middle_line[:, 0]  # 中线误差  （800 = 1600/2）



    ## 绘制拟合的 左 右边线 和 中线
    # for j in range(len(fit_x)):
    #     cv2.circle(img, (left_fit_y.astype(int)[j], fit_x.astype(int)[j]), 2, (255, 0, 255), 2)  # left_line
    #     cv2.circle(img, (right_fit_y.astype(int)[j], fit_x.astype(int)[j]), 2, (125, 0, 255), 2)  # right_line
    #     cv2.circle(img, (int(middle_line[j, 0]), int(middle_line[j, 1])), 2, (25, 0, 255), 2)  # middle_line

    #     cv2.circle(img, (int(mid_width), int(middle_line[j, 1])), 2, (0, 125, 255), 2)
    
    error_val = np.sum(
        np.multiply((middle_line[:, 0] - 800), (weight_matrix / np.sum(weight_matrix)))
        )  # 归一化误差
    rospy.loginfo('归一化误差：%f'%error_val)
    return error_val



if __name__ == "__main__":
    rospy.init_node('controler_node',anonymous=True)
    loop_rate=rospy.Rate(30)#10hz

    velocity_puber=rospy.Publisher('/mickrobot/chassis/cmd_vel',Twist,queue_size=1)#控制发布 进行速度控制
    Controler=PID_Controler()#pid控制器
    vel_msg=Twist()#速度消息

    Controler.velocity_pub(velocity_puber,
                               vel_msg,
                               forward_speed_x=0,
                               rotate_speed_z=0)#初始化速度设为0
    rospy.loginfo("cmd_vel发布初始化")

    args = get_args()
    # cap = cv2.VideoCapture(args.video_path)
    cap = cv2.VideoCapture(2)
    cap.set(3,1280)
    cap.set(4,720)
    #图像尺寸获取
    # img_high=320 
    # img_width=1600
    # mid_width=img_high/2

    #终端参数读取
    args = get_args()

    # cap = cv2.VideoCapture(args.video_path)
    # out = cv2.VideoWriter('./output2 .mp4', cv2.VideoWriter_fourcc(*"mp4v"),30, (1600, 903))
    # cap = cv2.VideoCapture(0)
    # cap.set(3,1920)
    # cap.set(4,1080)
    

    isnet = UFLDv2(args.engine_path, args.config_path, args.ori_size)


    weight = [0.1, 0.2, 0.4, 0.2, 0.1]  # 区域偏差权重 sum=1
    while not rospy.is_shutdown():
        # success, img = cap.read()
        img=cv2.imread('/home/yuyang/2.jpg')
        img = cv2.resize(img, (1600, 903))
        img = img[380:700, :, :]
        img_high=img.shape[0] 
        img_width=img.shape[1]
        mid_width=img_width/2
        # print(img.shape[0],img.shape[1])
        coords=isnet.forward(img)# 获取车道点
        # print(coords)
        if len(coords):
            error_val=Lane_fitting(coords,weight)
            print(error_val)
            if error_val == None:#没有左右边线就再次采样
                continue
        else:
            continue#继续扫图

        output_rotate_speed=Controler.output_control(error_val)  # 输出值
        print('asss',output_rotate_speed)
        rospy.loginfo("转向速度：%f"%output_rotate_speed)

        Controler.velocity_pub(velocity_puber,
                                   vel_msg,
                                   forward_speed_x = 0,
                                   rotate_speed_z = 0)
        rospy.loginfo("速度发布成功")
        rospy.loginfo("****************************************************")

        loop_rate.sleep()
    # out.release() #退出编辑 保存视频
