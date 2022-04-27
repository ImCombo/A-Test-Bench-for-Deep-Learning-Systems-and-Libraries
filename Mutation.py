import cv2
import numpy as np
import glob, os
import math
from matplotlib import pyplot as plt
from Functions import *
import time
import datetime

def gen_bkg_labels(id, labels, write_path):
    
    # save labels for other hands in background
      
    for cnt in range(len(labels)):
        filename = id[:-4] + "-" + str(cnt) + "BKG.txt"
        filepath = os.path.join(write_path+'BKG', filename)
        
        f = open(filepath, "w")
        for i in range(len(labels)):
            if i!=cnt:
                temp = ""
                for value in labels[i]:
                    temp = temp + str(value) + " "
                f.write(temp+'\n')
        f.close()

'''
object_removal generate single-object-removed mutated images 

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
'''

def object_removal(image_path, label_path, write_path):
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    
    os.chdir(write_path)
    if not os.path.exists('BKG'):
        os.mkdir('BKG') 
        
    image = 0
    start = time.time()

    for id in label_list:     
        image += 1
        if image % 1000 == 0:
            print("Done",image,"images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        
        #generate labels 
        gen_bkg_labels(id, labels, write_path)
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  
        img = cv2.imread(image_path+id[:-4]+".jpg")
        
        cnt = 0
        for box in bbox:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop_img = img.copy()
            for i in range(h):
                for j in range(w):
                    crop_img[y+i][x+j] = [0, 0, 0]

            cv2.imwrite('BKG/' + id[:-4] + "-"+ str(cnt) + "BKG.jpg", crop_img)      #save image
          
            cnt+=1
        
            
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")


def gen_obj_labels(id, labels, write_path, folder, ext):
       
    cnt = 0
    for label in labels:
        filename = id[:-4] + "-" + str(cnt) + ext + ".txt"
        filepath = os.path.join(write_path+folder, filename)
        f = open(filepath, "w")
        
        temp = ""
        for value in label:
            temp = temp + str(value) + " "
        f.write(temp)
        f.close()
        cnt +=1
        

'''
background_removal generate background-removed mutated images 

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
'''
def background_removal(image_path, label_path, write_path):
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    
    os.chdir(write_path)
    if not os.path.exists('OBJ'):
        os.mkdir('OBJ') 
        
    start = time.time()
    image = 0
    for id in label_list:
        image += 1
        if image % 1000 == 0:
            print("Done",image, "images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        
        #generate labels 
        gen_obj_labels(id, labels, write_path, 'OBJ', 'OBJ')
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  
        img = cv2.imread(image_path+id[:-4]+".jpg")
        
        #generate black background
        bg = np.uint8(0 * np.ones((img_h, img_w, 3)))       

        cnt = 0
        for box in bbox:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            obj = bg.copy()
            
            for i in range(h):
                for j in range(w):
                    obj[y+i][x+j] = img[y+i][x+j]

            cv2.imwrite('OBJ/' + id[:-4] + "-"+ str(cnt) + "OBJ.jpg", obj)      #save image
            cnt+=1
            
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")


'''
background_removal_5px generate background-removed mutated images, which 5px of area 
around the bounding box is retained 

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
'''
def background_removal_5px(image_path, label_path, write_path):
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    
    os.chdir(write_path)
    if not os.path.exists('OBJ_5px'):
        os.mkdir('OBJ_5px') 
        
    start = time.time()
    image = 0
    for id in label_list:
        image += 1
        if image % 1000 == 0:
            print("Done",image, "images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        
        #generate labels 
        gen_obj_labels(id, labels, write_path, 'OBJ_5px', 'OBJ_5px')
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  
        img = cv2.imread(image_path+id[:-4]+".jpg")
        
        #generate black background
        bg = np.uint8(0 * np.ones((img_h, img_w, 3)))       

        cnt = 0
        for box in bbox:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            obj = bg.copy()

            thickness = 5
            x = 0 if x - thickness < 0 else x - thickness
            y = 0 if y - thickness < 0 else y - thickness
            w = img_w - x if x + w + thickness*2>=img_w else w + thickness*2
            h = img_h - y if y + h + thickness*2>=img_h else h + thickness*2
                
            for i in range(h):
                for j in range(w):
                    obj[y+i][x+j] = img[y+i][x+j]
            
            cv2.imwrite('OBJ_5px/' + id[:-4] + "-"+ str(cnt) + "OBJ_5px.jpg", obj)      #save image
            cnt+=1
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")


'''
background_removal_int generate background-removed mutated images, which the background is dimmed with a factor,
instead of fully blackened

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
factor:     factor for multiplying the background RGB values. Default is 0.1
'''
def background_removal_int(image_path, label_path, write_path, factor=0.1):
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    
    os.chdir(write_path)
    if not os.path.exists('OBJ_INTENSITY_'+str(factor)):
        os.mkdir('OBJ_INTENSITY_'+str(factor)) 
    
    start = time.time()
    image = 0
    for id in label_list:
        image += 1
        if image % 1000 == 0:
            print("Done",image, "images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        
        #generate labels 
        gen_obj_labels(id, labels, write_path,'OBJ_INTENSITY_'+str(factor), 'OBJ_INT_'+str(factor))
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  
        img = cv2.imread(image_path+id[:-4]+".jpg")
        #generate black background
        #bg = np.array(img)
        bg = img.copy()
        bg = bg*factor

        cnt = 0
        for box in bbox:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            obj = bg.copy()
            
            for i in range(h):
                for j in range(w):
                    obj[y+i][x+j] = img[y+i][x+j]

            cv2.imwrite('OBJ_INTENSITY_'+str(factor) +'/' + id[:-4] + "-"+ str(cnt) + "OBJ_INT_" +str(factor) +".jpg", obj)      #save image
            cnt+=1
            
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")

def gen_bwo_labels(id, labels, write_path):      
    cnt = 0
    for label in labels:
        filename = id[:-4] + "-" + str(cnt) + "BwO.txt"
        filepath = os.path.join(write_path+'BwO', filename)
        f = open(filepath, "w")
        
        temp = ""
        for value in label:
            temp = temp + str(value) + " "
        f.write(temp)
        f.close()
        cnt +=1

'''
background_with_object generate mutated images of the background with one object

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
'''
def background_with_object(image_path, label_path, write_path):
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    
    os.chdir(write_path)
    if not os.path.exists('BwO'):
        os.mkdir('BwO') 
        
    start = time.time()
    image = 0
    
    for id in label_list:
        image += 1
        if image % 1000 == 0:
            print("Done",image, "images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        
        #generate labels
        gen_bwo_labels(id, labels, write_path)
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  
        img = cv2.imread(image_path+id[:-4]+".jpg")

        for obj in range(len(bbox)):
            crop_img = img.copy()
            for cnt in range(len(bbox)):
                if obj==cnt: continue

                x, y, w, h = int(bbox[cnt][0]), int(bbox[cnt][1]), int(bbox[cnt][2]), int(bbox[cnt][3]) 
                for i in range(h):
                    for j in range(w):
                        crop_img[y+i][x+j] = [0, 0, 0]

            cv2.imwrite("BwO/" + id[:-4] + "-"+ str(obj) + "BwO.jpg", crop_img)      #save image
            
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")

def gen_ins_bwo_labels(id, labels, write_path, cnt):      
    label = labels[cnt]
    filename = id[:-9] + "-" + str(cnt) + "INS_BwO.txt"
    filepath = os.path.join(write_path+'INS_BwO', filename)
    f = open(filepath, "w")

    temp = ""
    for value in label:
        temp = temp + str(value) + " "
    f.write(temp)
    f.close()
    cnt +=1

'''
background_with_object_ins mutated the object-inserted images to generate the background with one object images

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images


*the object-inserted images to be mutate should be generated by our functions
'''
def background_with_object_ins(image_path, label_path, write_path):
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    
    os.chdir(write_path)
    if not os.path.exists('INS_BwO'):
        os.mkdir('INS_BwO') 
        
    start = time.time()
    image = 0
    
    for id in label_list:
        image += 1
        if image % 1000 == 0:
            print("Done",image, "images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  
        img = cv2.imread(image_path+id[:-4]+".jpg")
        crop_img = img.copy()
        
        for cnt in range(len(bbox)):
            #if str(cnt) == id[9:10]: 
            curObj = id.find('-') +1

            if(curObj == 0 or not id[curObj:curObj+1].isdigit()):
                print("cannot find object in", id)
                return
            
            if str(cnt) == id[curObj:curObj+1]:
                gen_ins_bwo_labels(id, labels, write_path, cnt)
                continue

            x, y, w, h = int(bbox[cnt][0]), int(bbox[cnt][1]), int(bbox[cnt][2]), int(bbox[cnt][3]) 
            for i in range(h):
                for j in range(w):
                    crop_img[y+i][x+j] = [0, 0, 0]

        cv2.imwrite("INS_BwO/" + id[:-curObj] + "-"+ id[curObj:curObj+1] + "INS_BwO.jpg", crop_img)      #save image
        
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")


def gen_b_labels(id, write_path):
    filepath = os.path.join(write_path+'B', id[:-4] + "-B.txt")
    f = open(filepath, "w")
    f.close()


'''
multi_object_removal removed all objects in the images, generating images of only background

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
'''

def multi_object_removal(image_path, label_path, write_path):
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    
    os.chdir(write_path)
    if not os.path.exists('B'):
        os.mkdir('B') 
        
    start = time.time()
    image = 0
    
    for id in label_list:
        image += 1
        if image % 1000 == 0:
            print("Done",image, "images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        
        #generate labels
        gen_b_labels(id, write_path)
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  
        img = cv2.imread(image_path+id[:-4]+".jpg")

        crop_img = img.copy()
        for box in bbox:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3]) 
            for i in range(h):
                for j in range(w):
                    crop_img[y+i][x+j] = [0, 0, 0]
                    
        #save image          
        cv2.imwrite("B/" + id[:-4] + "-B.jpg", crop_img)    
        
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")

#use this if the inserted object is the same as the label
def gen_ins_label(id, labels, new_labels, write_path):
    
    for cnt in range(len(new_labels)):
        original_labels = labels.copy()
        filename = id + "-" + str(cnt) + "INS.txt"
        filepath = os.path.join(write_path+"INS", filename)
        f = open(filepath, "w")        
        original_labels.append(new_labels[cnt])
        for label in original_labels:
            temp=""
            for value in label:
                temp = temp + str(value) + " "
            f.write(temp+'\n')
        f.close()


#use this if the inserted object is the same as the label
def gen_nonhand_label(id, labels, write_path):
    
    for cnt in range(len(labels)):

        filename = id + "-" + str(cnt) + "INS.txt"
        filepath = os.path.join(write_path+"INS", filename)
        f = open(filepath, "w")        

        for label in labels:
            temp=""
            for value in label:
                temp = temp + str(value) + " "
            f.write(temp+'\n')
        f.close()


def isOverlap(label, pred):
    x1 = label[0]
    y1 = label[1]
    w1 = label[2]
    h1 = label[3]
    
    x2 = pred[0]
    y2 = pred[1]
    w2 = pred[2]
    h2 = pred[3]
    
    XA1 = x1
    XA2 = x1 + w1
    YA1 = y1
    YA2 = y1 + h1
    XB1 = x2
    XB2 = x2 + w2
    YB1 = y2
    YB2 = y2 + h2
    
    SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    if (SI>0): 
      return True
    else: 
      return False


def find_pos(bbox, tar, hand_size, x,y,w,h, buffer, WIDTH, HEIGHT):
       
    #find inserted hand size
    k = math.sqrt(hand_size/12)
    hand_w, hand_h = int(3*k), int(4*k)
    #hand_w, hand_h = cal_hand_size(hand_size)
    
    #right, left, down, up, RU, RD, LD, LU
    dx = [w+buffer, -hand_w-buffer, 0, 0, w+buffer, w+buffer, -hand_w-buffer, -hand_w-buffer]         
    dy = [0, 0, h+buffer, -hand_h-buffer, -hand_h-buffer, h+buffer, h+buffer, -hand_h-buffer]
    
    pos = 0
    found = False   
    while pos<=7 and not found:
        overlap = False
        out = False

        #check out of bound
        if ((x+dx[pos]<0) or (x+dx[pos]+hand_w>=WIDTH) or (y+dy[pos]+hand_h>=HEIGHT) or (y+dy[pos]<0)):
            pos+=1
            out = True
            continue

        for other in range(len(bbox)):
            if (tar == other): continue;  

            check = [x+dx[pos], y+dy[pos], hand_w, hand_h]     #inserted hand     

            if(isOverlap(check, bbox[other])):              #check if inserted hand overlap with other hand
                pos+=1
                overlap = True
                break        
        if not overlap and not out: 
            found = True
    return dx, dy, hand_w, hand_h, pos



def insertobj(image_path, filename, bbox, object_path):
    img = cv2.imread(image_path+filename)
    obj_img = cv2.imread(object_path, -1)
    
    img_w, img_h = img.shape[1], img.shape[0]

    insert_label = []   

    for tar in range(len(bbox)):
        img2 = img.copy()
        x, y, w, h = int(bbox[tar][0]), int(bbox[tar][1]), int(bbox[tar][2]), int(bbox[tar][3])

        area = w*h
        buffer = 0
        if area<800:
            area = 800
            
        dx,dy,hand_w, hand_h, pos = find_pos(bbox, tar, area, x,y,w,h, buffer, img_w, img_h)
        while(pos>7):
            print("Fail to insert hand ", tar," in ", filename,"! Try smaller hand")     

            #if all position fails, decrease hand size
            #if still fail, move away from the obj 
            if(area>500):
                area = area-200
            elif (area>100):
                area = area-50
            else:
                area = w*h
                buffer = buffer + 10
                print("Move further")

            dx,dy, hand_w, hand_h, pos = find_pos(bbox, tar, area, x,y,w,h, buffer, img_w, img_h)

        dim = (hand_w, hand_h)
        resized = cv2.resize(obj_img, dim)
        for i in range(hand_h):
              for j in range(hand_w):
                weight = resized[i][j][-1:]/255.0

                img2[i+y+dy[pos]][j+x+dx[pos]] = (resized[i][j][:3]*weight+img2[i+y+dy[pos]][j+x+dx[pos]]*(1-weight)).astype(int)

        new_label = [x+dx[pos], y+dy[pos], hand_w, hand_h]

        insert_label.append(normalize(new_label, img_w, img_h))
        '''
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img2)
        '''
        cv2.imwrite("INS/" + filename[:-4] + "-"+ str(tar) + "INS" + ".jpg", img2)
    return insert_label


'''
object_insertion generate mutated images by inserting a non-hand object

image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
object_path: path to the image of the object to be inserted
'''
def object_insertion(image_path, label_path, write_path, object_path):
    #os.chdir(label_path)
    #label_list = glob.glob("*.txt")
    label_list = [file for file in os.listdir(label_path) if file.endswith(".txt")]
    os.chdir(write_path)
    if not os.path.exists(write_path + "/INS"):
        os.mkdir(write_path + "/INS") 
        
    start = time.time()
    image = 0
    
    for id in label_list:
        image += 1
        if image % 1000 == 0:
            print("Done",image, "images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
    
        
        #mutate images 
        img_w, img_h = get_wh(image_path+id[:-4]+".jpg")
        bbox = unnormalize(labels, img_w, img_h)  

        new_labels = insertobj(image_path, id[:-4]+".jpg", bbox, object_path)
        #gen_ins_label(id[:-4], labels, new_labels, write_path)
        gen_nonhand_label(id[:-4], labels, write_path)
        
    print("Finish mutating", image,"images, time passed: ", round(time.time()- start,2), "s")


def log(s):
    s = str(s)
    f = open('log.txt', 'a')
    f.write(s+'\n')
    f.close()

    
# This function generates the RESIZE set, each object is enlarged up to size within the boundary and placed in the center.
'''
object_insertion generate mutated images by inserting a non-hand object
image_path: path to the directory that stores the images to be mutated
label_path: path to the directory that stores the labels of the images. 
            Each label should has the same name as its associated image
write_path: path to the directory that stores the mutated images
sizes: array of float containing the result sizes of the object, e.g. [0.05], [0.05, 0.10]
'''

def object_resize(image_path, label_path, write_path, sizes):
    # check input and create path accordingly
    log('----------------------------------------')
    log(str(datetime.datetime.now())[5:10])
    log('start object_resize')
    if not os.path.exists(image_path):
        log('image path does not exist')
        return
    if not os.path.exists(label_path):
        log('label path does not exist')
        return
    os.chdir(label_path)
    label_list = glob.glob("*.txt")
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    os.chdir(write_path)
    log('image path: '+image_path)
    log('label path: '+label_path)
    log('write path: '+write_path)
    log('sizes: '+str(sizes))
    if (len(sizes) == 0):
        log('No sizes given')
        return
    for size in sizes:
        if not os.path.exists('OBJ_RESIZE_'+str(size)):
            os.mkdir('OBJ_RESIZE_'+str(size)) 
   
    # start mutation
    num_image = 0
    start = time.time()
    for id in label_list:
        num_image += 1
        if num_image % 1000 == 0:
            log("Done "+str(num_image)+" images")
            
        labels = get_label(id, label_path)
        if not len(labels):
            continue
        fileid = id[:-4]
        img = cv2.imread(image_path+fileid+".jpg")
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        
        if len(labels) != 0:
            cnt = 0
            for label in labels:
                box = unnormalize([label], img_w, img_h)[0]
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                for size in sizes:
                    temp_img = np.zeros((img_h, img_w, 3), np.uint8)
                    crop_img = img[y:y+h, x:x+w]
                    obj_w = crop_img.shape[1]
                    obj_h = crop_img.shape[0]
                    obj_area = obj_w * obj_h
                    
                    # resize object
                    ratio = obj_area / (img_area)
                    if (ratio < size):
                        upscale = math.sqrt(size/ratio)
                        new_w = int(crop_img.shape[1] * upscale)
                        new_h = int(crop_img.shape[0] * upscale)
                        if (new_w >= img_w):
                            downscale = img_w / new_w
                            new_w = img_w
                            new_h = int(new_h * downscale)
                        if (new_h >= img_h):
                            downscale = img_h / new_h
                            new_h = img_h
                            new_w = int(new_w * downscale)
                        dim = (new_w, new_h)
                        crop_img = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
                        
                    # put resized object into new image
                    new_w, new_h = crop_img.shape[1], crop_img.shape[0]
                    new_x, new_y = int((img_w-new_w)/2), int((img_h-new_h)/2)
                    for i in range(new_h):
                        temp_img[new_y+i][new_x:new_x+new_w] = crop_img[i]
                    crop_img_area = crop_img.shape[0]*crop_img.shape[1]
                    temp_img_area = temp_img.shape[0]*temp_img.shape[1]
                    
                    cv2.imwrite("OBJ_RESIZE_" + str(size) + '/'+ fileid + "-"+ str(cnt) + "OBJ_RESIZE" + ".jpg", temp_img)      #save image
                    f = open("OBJ_RESIZE_" + str(size) + '/'+ fileid + "-"+ str(cnt) + "OBJ_RESIZE" + ".txt", 'w')
                    f.write(str(label[0])+' '+str((new_x+new_w/2)/img_w)+' '+str((new_y+new_h/2)/img_h)+' '+str(new_w/img_w)+' '+str(new_h/img_h)+'\n')
                    f.close()
                cnt+=1
    log("Finish resize "+str(num_image)+" images with sizes = "+str(sizes)+", time passed: "+str(round(time.time()- start,2))+" s")

