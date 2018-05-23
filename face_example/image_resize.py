#!/usr/bin/python  
# -*- coding:utf-8 -*-  
  

import os  
import shutil  
from PIL import Image   
import sys 
 
processIndex = 0  
'''
@brief: 递归遍历文件夹并进行图片缩放
@param:
    imgPath: the top folder of the imgs.
    to_scale: scale_size, num of double type.
'''
def resizeImg(imgPath, to_scale):  
    global processIndex  
    fileList = []  
    files = os.listdir(imgPath)  
    for f in files:  
        filePath = imgPath + os.sep + f  
        if(os.path.isfile(filePath)):  
            fileList.append(f)  
        elif(os.path.isdir(filePath)):  
            resizeImg(filePath, to_scale)  
    for fileName in fileList:  
        processIndex += 1  
        fileFullName = imgPath + os.sep + fileName  
        suffix = fileName[fileName.rfind('.'):]  
        if(suffix == '.png' or suffix == '.jpg'):  
            print 'processing the ' + str(processIndex) + 'th file:' + fileFullName  
            img = Image.open(fileFullName)  
            w,h = img.size   
            tw = int(w * to_scale)  
            th = int(h * to_scale)  
            reImg = img.resize((tw,th),Image.ANTIALIAS)  
            reImg.save(fileFullName)  
            del reImg  
            
if __name__ == '__main__': 
    if len(sys.argv) != 3:
        print'Usage:python %s original_folder scaled_folder'%(sys.argv[0])
        sys.exit() 
        
    original_folder = sys.argv[1]
    scaled_folder = sys.argv[2]
    if os.path.isdir(scaled_folder):  
        flag = raw_input('the output dir is exist, sure to del it ? :(y/n)')  
        if flag == 'y' or flag == 'yes' or flag == 'Y' or flag == 'YES':  
            try:     
                shutil.rmtree(scaled_folder)  
            except Exception, e:
                print 'remove failed !',Ecxeption,":", e
        else:  
            exit  
    shutil.copytree(original_folder, scaled_folder)  
    scale_size = 2;
    resizeImg(scaled_folder, scale_size)  
    print "resize success." 