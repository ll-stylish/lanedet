#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_msgs.msg import String
from cv_bridge import CvBridge , CvBridgeError
import cv2
import time
import random

def Cam_img_pub():
    capure=cv2.VideoCapture(0)
    capure.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
    capure.set(3,1280)
    capure.set(4,720)

    
    cv_bridge=CvBridge()
    rospy.init_node('camera_node',anonymous=True)
    img_publisher=rospy.Publisher('/image_view/image_raw',Image,queue_size=1) # topic
    while not rospy.is_shutdown():
        #time_now=time.time()
        ret,frame = capure.read()#设置摄像头
        
        
        #如果有图像
        if ret:
            # print(capure.get(5))#获取；帧率
            frame=cv2.resize(frame,(640,360))

            img_publisher.publish(cv_bridge.cv2_to_imgmsg(frame,'bgr8'))
            # rospy.loginfo('成功发送一张图片')
            # print(time.time()-time_now)
            
            #测试图片效果
            # cv2.imwrite('/home/yuyang/桌面/file/imgs/img'+str(random.randint(0,100))+'.jpg',frame)
    capure.release()
    cv2.destroyAllWindows()
            

if __name__ == '__main__':
    # rospy.loginfo("%s"%rospy.get_param("cam_topic_param"))
    # pic_sub = rospy.Subscriber(rospy.get_param("cam_topic_param"),Image)

    Cam_img_pub()

    