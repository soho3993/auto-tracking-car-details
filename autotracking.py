# coding: utf-8
import os
import cv2
import numpy as np
import random
import time
from PIL import Image, ImageEnhance

#两个像素是否误匹配
def ifsame(pt1, pt2, bbox):
    x1 = (pt1[0]-bbox[0])/(bbox[2] - bbox[0])          #x, y是点的横/纵坐标与bbox的长/宽的比值
    y1 = (pt1[1]-bbox[1])/(bbox[3]  - bbox[1])
    x2 = (pt2[0]-bbox[0])/(bbox[2] - bbox[0])
    y2 = (pt2[1]-bbox[1])/(bbox[3]  - bbox[1])
    if abs(x1-x2) < 0.02 and abs(y1 - y2) < 0.02:      #如果比值大于bbox的0.02，则认为是误匹配
        return True
    else:
        return False

#将ROI宽高比强制压缩为1-2.3之间
def fixbbox(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (bbox[0] + bbox[2])/2
    y = (bbox[1] + bbox[3])/2
    if w/h > 2.3:
        ret = (x-h/2*2.3, bbox[1], x+h/2*2.3, bbox[3])
    elif w/h < 1:
        ret = (bbox[0], y-w/2*1.0, bbox[2], y+w/2*1.0)
    else:
        ret = bbox
    return ret

#ROI长宽比置为1
def fixbbox2(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    a = min(w,h)/2
    x = (bbox[0] + bbox[2])/2
    y = (bbox[1] + bbox[3])/2
    ret = (x-a, y-a, x+a, y+a)
    return ret

#统计法排除误跟踪点（后向）
def getedge(a):
    a.sort()
    le = int(len(a)*0.05)
    ri = int(len(a)*0.95)
    l = a[le]
    r = a[ri]
    for i in range(len(a)):
        if abs(a[i] - (l+r)/2)/((l+r)/2) < 0.1:
            if i < le:
                le = i
            if i > ri:
                ri = i
    b = a[le:ri+1]
    c = np.array(b)
    m = c.mean()
    std = c.std(ddof = 1)
    return max(a[le],m - 3*std), min(a[ri],m+3*std)

#统计法排除误跟踪点（前向）
def getedge2(a):
    a.sort()
    c = np.array(a)
    m = c.mean()
    std = c.std(ddof = 1)
    return max(a[0],m - 3*std), min(a[-1],m+3*std)

#滑动平均
def getbbox(i,bboxold):
    cnt = 3
    b = bboxold[i-1]
    c = bboxold[i]
    a = bboxold[i+1]
    if b == (0,0,0,0) or c == (0,0,0,0) or a == (0,0,0,0):
        return (0,0,0,0)
    else:
        l = (b[0] + c[0] + a[0])/cnt
        u = (b[1] + c[1] + a[1])/cnt
        r = (b[2] + c[2] + a[2])/cnt
        d = (b[3] + c[3] + a[3])/cnt
        return (l,u,r,d)

#判断前向是否移动到屏幕边缘
def ifedge(pt, w, h):
    if pt[0] < w*0.05 or pt[0] > w* 0.95 or pt[1] < h * 0.05 or pt[1] > h * 0.95:
        return True
    else:
        return False

#得到bbox的中心点
def getcenter(left, up, right, down):
    w = (left + right)/2
    h = (up + down)/2
    return(w, h)

#前向后向ROI合并
def addbbox(bbox0, bbox1, r):
    if bbox0 == 0 and bbox1 == 0:
        return (0,0,0,0)
    if bbox0 == 0:
        return bbox1
    if bbox1 == 0:
        return bbox0
    return ((bbox0[0]*(1-r) + bbox1[0]*r), (bbox0[1]*(1-r) + bbox1[1]*r), (bbox0[2]*(1-r) + bbox1[2]*r), (bbox0[3]*(1-r) + bbox1[3]*r))

#缩放ROI
def trbbox(bbox, wr, hr):
    x, y = getcenter(bbox[0], bbox[1], bbox[2], bbox[3])
    w = (bbox[2] - bbox[0])/wr/2
    h = (bbox[3] - bbox[1])/hr/2
    return (x-w, y-h, x+w, y+h)

#后向跟踪
def backprop(cache):
    w = 640
    h = 360
    framen = len(cache)-1                                                                           #待读取的帧数（因为要比较当前帧与下一帧所以减1）

    bbox0 = (0,0,w,h)                                                                               #初始ROI（整幅画面）
    bboxb = [(bbox0[0]/w*size[0], bbox0[1]*size[1]/h,bbox0[2]*size[0]/w, bbox0[3]*size[1]/h)]       #用来存储每一帧的ROI用于判准
    skip = False
    for i in range(framen):
        if not skip:                                                                                #如果不跳过，更新前一帧的画面
            #当前帧im0
            name0 = cache[i]
            img0 = cv2.cvtColor(name0,cv2.COLOR_BGR2GRAY)
            im0 = cv2.resize(img0,(w,h))                                                            #缩小图像加快处理速度
        #下一帧im
        name = cache[i+1]
        img = cv2.cvtColor(name,cv2.COLOR_BGR2GRAY)
        im = cv2.resize(img,(w,h))

        if i == 0:                                                                                  #视频第一帧，初始化跟踪点pt0
            pt0 = cv2.goodFeaturesToTrack(im0, 500, 0.01, int(w/50))
        elif not skip:                                                                              #以后直接迭代
            pt0 = pt

        #得到下一帧的跟踪点位置pt
        pt = np.empty(len(pt0))
        res = cv2.calcOpticalFlowPyrLK(im0, im,pt0,pt)
        pt = res[0]

        pt1 = []                                                                                    #跟踪点筛选后的list

        x = []
        y = []

        for j in range(len(pt)):
            if res[1][j][0] == 1 and ifsame(pt0[j][0],pt[j][0], (0,0,w,h)):                         #下一帧的跟踪点是否误匹配
                pt1 += [[[pt[j][0][0], pt[j][0][1]]]]                                               #将正确的点加入下一次迭代
                
                #每个点的横纵坐标
                x += [pt[j][0][0]]
                y += [pt[j][0][1]]

        if len(pt1) < 50:                                                                           #如果追踪到的点减少到50个以下，跳过该帧，直接进行下一帧
            skip = True
        else:
            skip = False
            
        if not skip:
            #根据xy坐标得到soft margin bbox
            lefttest, righttest = getedge(x)
            uptest, downtest = getedge(y)

            pt2 = np.array(pt1)                                                                    #list转np.array
            pt = pt2

            bbox1 = (lefttest, uptest, righttest, downtest)                                        #下一帧的ROI
            bbox0 = fixbbox(bbox1)                                                                 #ROI长宽比出格时修正ROI，并更新ROI
            
            bboxb += [(bbox0[0]/w*size[0], bbox0[1]*size[1]/h,bbox0[2]*size[0]/w, bbox0[3]*size[1]/h)]    #加入历史ROIlist
            
        else:
            bboxb += [0]                                                                           #如果追踪失败，ROI以0占位
    bboxb[0] = bboxb[5]
    bboxb[1] = bboxb[5]
    bboxb[2] = bboxb[5]
    bboxb[3] = bboxb[5]
    bboxb[4] = bboxb[5]
    return bboxb

#前向跟踪
def frontprop(cache):
    fNUMS = len(cache)
    framen = fNUMS  - 1
    pts = []                                                                                      #存每帧的点的index和点的坐标
    bboxf = [0]*fNUMS
    skip = False
    for i in range(framen):
        if not skip:
            name0 = cache[framen-i]
            img0 = cv2.cvtColor(name0,cv2.COLOR_BGR2GRAY)
        name = cache[framen-i-1]
        img = cv2.cvtColor(name,cv2.COLOR_BGR2GRAY)

        if i == 0:
            pt0 = cv2.goodFeaturesToTrack(img0, 5000, 0.01, size[0]/200)                         #第一帧生成初始待跟踪的点
            res0 = [1] * len(pt0)                                                                #待跟踪点的状态（全部存在）
            ptt = []                                                                             #ptt是前一帧还在的点的index和点的坐标
            for j in range(len(pt0)):
                ptt += [[j, pt0[j]]]
                
        pts += [ptt]

        #从ptt取出pt0
        pt00 = []
        pt0index = []
        if not ptt == -1:
            for j in range(len(ptt)):
                if res0[ptt[j][0]] == 1:
                    pt0index += [ptt[j][0]]          #点的index
                    pt00 += [ptt[j][1]]              #点的坐标
            pt0 = np.array(pt00)
    
        if len(pt0) < 5:
            skip = True
        else:
            skip = False
        
        if not skip:
            #pt0是传进来的还在的点的坐标
            #得到下一帧的跟踪点位置pt
            pt = np.empty(len(pt0))
            res = cv2.calcOpticalFlowPyrLK(img0, img, pt0, pt)
            pt = res[0]

            ptt = []                                                                             #跟踪点筛选后的list（点的index和点的坐标）
            #判断每个留下的点，如果是ROI内的，置1，如果是ROI外的，置0.5，其他噪声点：置0
            for j in range(len(pt)):
                if res[1][j][0] == 1 and ifsame(pt0[j][0],pt[j][0], (0,0,size[0],size[1])):      #下一帧的跟踪点是否误匹配
                    ptt += [[pt0index[j], [[pt[j][0][0], pt[j][0][1]]]]]                         #将还在的点的index和坐标的点加入下一次迭代

                    if ifedge(pt[j][0],size[0],size[1]):
                        res0[pt0index[j]] = 0.5
                else:
                    res0[pt0index[j]] = 0
        else:
            ptt = -1
            
    #接下来生成bbox
    for i in range(len(pts)):
        pt_todraw = pts[i]
        if not pt_todraw == -1:
            x = []
            y = []
            for pt in pt_todraw:
                if res0[pt[0]] == 1:
                    x += [pt[1][0][0]]
                    y += [pt[1][0][1]]
            if len(x) > 2:
                lefttest, righttest = getedge2(x)
                uptest, downtest = getedge2(y)
                bbox1 = (lefttest, uptest, righttest, downtest)                                  #下一帧的ROI
                bbox0 = fixbbox(bbox1)                                                           #ROI长宽比出格时修正ROI，并更新ROI
                bboxf[framen - i]  = bbox0                
    return bboxf

#将前向和后向生成的ROI合并
def combinebbox(bboxb, bboxf, shrink_rate):
    fNUMS = len(bboxb)
    bboxlist = []                                                                                #用于存放最终的ROI列表
    for i in range(fNUMS):
        bbox0 = bboxb[i]
        bbox1 = bboxf[i]
        rat = i/fNUMS
        bboxlist += [addbbox(bbox0, bbox1, rat)]                                                 #将前向和后向的bbox合并
    wr = 1                                                                                       #横向和纵向的放缩比例
    hr = 1
    trust = True
    skip = False
    havebbox1 = False
    for bbox in bboxlist:
        if not bbox == (0,0,0,0):
            bbox1 = bbox
            havebbox1 = True
            break
    if not havebbox1:
        bbox1 = (0,0,1920,1080)
    for i in range(fNUMS - 1):
        if not skip:
            bbox0 = bbox1
        bbox1 = bboxlist[i+1]
        if bbox1 == (0,0,0,0):                                                                   #如果下一帧没有ROI，则跳过
            skip = True
        else:
            skip = False
        if not skip:
            if abs((bbox1[2] - bbox1[0])/(bbox0[2] - bbox0[0])/wr - 1) > 0.1:                    #如果两帧之间宽度之比小于0.9或大于1.1
                wr = (bbox1[2] - bbox1[0])/(bbox0[2] - bbox0[0])*shrink_rate                     #强行置为前一帧的shrink_rate倍
            if abs((bbox1[3] - bbox1[1])/(bbox0[3] - bbox0[1])/hr - 1) > 0.1:
                hr = (bbox1[3] - bbox1[1])/(bbox0[3] - bbox0[1])*shrink_rate
            bbox1 = trbbox(bbox1, wr, hr)
            x1, y1 = getcenter(bbox0[0],bbox0[1],bbox0[2],bbox0[3])
            x2, y2 = getcenter(bbox1[0],bbox1[1],bbox1[2],bbox1[3])
            if abs(x1-x2) > (bbox0[2] - bbox0[0])/3 or abs(y1-y2) > (bbox0[3] - bbox0[1])/3:     #如果中心点位移大于1/3则认为不可靠
                trust = False
            if trust == True:
                if (bbox1[2] - bbox1[0])/(bbox0[2] - bbox0[0]) > 1:                              #如果下一帧ROI比上一帧的大
                    wr1 = (bbox1[2] - bbox1[0])/(bbox0[2] - bbox0[0])/shrink_rate                #强行置为前一帧的shrink_rate倍
                else:
                    wr1 = 1
                if (bbox1[3] - bbox1[1])/(bbox0[3] - bbox0[1]) > 1:
                    hr1 = (bbox1[3] - bbox1[1])/(bbox0[3] - bbox0[1])/shrink_rate
                else:
                    hr1 = 1
                bbox1 = trbbox(bbox1, wr1, hr1)
                bboxlist[i+1] = bbox1
            else:
                bboxlist[i+1] = (0,0,0,0)
        else:                                                                                    #如果新一帧没有ROI则把前一帧的ROI缩小shrink_rate倍用于后续判准
            wr0 = 1/shrink_rate
            hr0 = 1/shrink_rate
            bbox0 = trbbox(bbox0, wr0, hr0)
    return bboxlist

#从生成的ROI中随机抽取20对
def getlst(bboxlist, ifsquare):
    fNUMS = len(bboxlist)
    outlst = []                                                                                  #用于存放随机抽取的数据
    mid = int((fNUMS/2)**1/2 * 10)
    failnum = 0
    fail = False
    while len(outlst) < 20:
        rd = random.randint(30,mid)
        rdn = (int(rd**2/100))
        try:
            dist =  rdn + random.randint(50,fNUMS - 10 - rdn)
        except:
            failnum += 1
        else:
            bbox1 = getbbox(rdn,bboxlist)
            bbox2 = getbbox(dist,bboxlist)
            if not bbox1 == (0,0,0,0) and not bbox2 == (0,0,0,0) and min((bbox1[2] - bbox1[0]), (bbox1[3] - bbox[1])) > 223:
                if ifsquare:
                    mat = [dist, fixbbox2(bbox2), rdn, fixbbox2(bbox1)]                          #远景在前近景在后
                else:
                    mat = [dist, bbox2, rdn, bbox1]
                if not mat in outlst:
                    outlst += [mat]
                else:
                    failnum += 1
            else:
                failnum += 1

        if failnum > 100000:
            fail = True
            break
    if fail == True:
        return fail, 0
    else:
        return fail, outlst

#从失败的视频中随机抽取20对
def getlst2(fNUMS):
    outlst = []                                                                                 #用于存放随机抽取的数据
    mid = int((fNUMS/2)**1/2 * 10)
    failnum = 0
    fail = False
    while len(outlst) < 20:
        rd = random.randint(30,mid)
        rdn = (int(rd**2/100))
        try:
            dist =  rdn + random.randint(50,fNUMS - 10 - rdn)
        except:
            failnum += 1
        else:
            mat = [dist, rdn]                                                                   #远景在前近景在后
            if not mat in outlst:
                outlst += [mat]
            else:
                failnum += 1
        if failnum > 100000:
            fail = True
            break
    if fail == True:
        return fail, 0
    else:
        return fail, outlst

#--------------------------------------------------主函数-----------------------------------------------------
hav = False
print('请输入源路径（默认路径为当前路径）：')
while not hav:
    time.sleep(0.2)
    src = input()
    if os.path.exists(src) or src == '':
        hav = True
        if src == '':
            src = os.path.abspath('.')
    else:
        print('该路径不存在！请重新输入：')
hav = False
print('请输入目标路径（默认路径为当前路径）：')
while not hav:
    time.sleep(0.2)
    dst = input()
    if os.path.exists(dst) or dst == '':
        hav = True
        if dst == '':
            dst = os.path.abspath('.')
    else:
        try:
            os.makedirs(dst)
        except:
            print('该路径不存在，且无法自动创建！')
        else:
            print('该路径不存在，已自动创建！')
            hav = True

print('是否将ROI置为正方形？（默认为否）')
time.sleep(0.2)
ifs = input('y or n:')
if ifs == 'y' or ifs == 'Y':
    ifsquare = True
else:
    ifsquare = False

path = src                                                 #源文件目录
rpath = dst                                                #目标文件目录
rev = True                                                 #是否反转视频：
#ifsquare = True                                           #是否将ROI置为正方形
shrink_rate = 0.995                                        #每帧压缩ROI的比率
lst = os.listdir(path)
lst.sort()

#如果指定输出路径不存在，则创建
if os.path.exists(rpath):                 
    pass
else:
    os.makedirs(rpath)
    print('目标路径不存在，已自动创建！')
    
#创建存储成功图片和视频的路径
path1 = rpath + '/success/picture/data1'                   #原图的路径
try:
    os.makedirs(path1)
except:
    pass
path2 = rpath + '/success/picture/data2'                   #有框的图的路径
try:
    os.makedirs(path2)
except:
    pass

path3 = rpath + '/success/video'                           #视频的路径
try:
    os.makedirs(path3)
except:
    pass

#创建存储失败图片和视频的路径
path4 = rpath + '/fail/picture'                            #错误视频图片路径
try:
    os.makedirs(path4)
except:
    pass
path5 = rpath + '/fail/video'                              #错误视频的路径
try:
    os.makedirs(path5)
except:
    pass

f = open(rpath + '/success/picture/pic_data.txt', "w+")    #用来存储图片信息

for k in range(len(lst)):                                  #待处理的视频编号
    video_name = lst[k]
    video_full_path = os.path.join(path, video_name)       #得到待处理视频的完整文件名
    videoCapture = cv2.VideoCapture(video_full_path)       #创建OpenCV的视频内容读取对象
    fps = videoCapture.get(cv2.CAP_PROP_FPS)                                                                    #视频帧率
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))   #视频像素
    fNUMS = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))                                                     #视频总帧数
    #w, h为后向传播时将图片压缩以加快运算速度
    w = 640
    h = 360

    #-----------------------读取每帧的图像并存入cache中----------------------------
    cache = []
    cache1 = []
    if fNUMS < 100:                                       #排除掉非视频以及帧数小于100的视频
        print(video_name, '未处理')
        continue
    for i in range(fNUMS):
        success, frame = videoCapture.read()
        cache1 += [frame]
        if not success:
            break
        a = Image.fromarray(np.uint8(frame))
        b = ImageEnhance.Sharpness(a).enhance(10.0)
        frame = np.array(b)
        cache += [frame]
    videoCapture.release()
    if rev:                                               #是否反转视频
        cache.reverse()
        cache1.reverse()
    #-----------------------------得到结果--------------------------------------
    bboxb = backprop(cache)                               #后向传播
    bboxf = frontprop(cache)                              #前向传播
    bboxlist = combinebbox(bboxb, bboxf, shrink_rate)     #ROI整合                                    
    

    #---------------------------------跟踪结果导出-------------------------------
    snum = 0                                              #得到ROI的帧数
    fail = False                                          #记录是否失败
    fNUMS = len(bboxlist)
    
    #计算成功跟踪的帧的比例
    for i in range(fNUMS):
        if i == 0 or i == fNUMS - 1:
            bbox = bboxlist[i]
        else:
            bbox = getbbox(i,bboxlist)
        if not bbox == (0,0,0,0):
            snum += 1
    if snum/fNUMS > 0.95:                                 #如果有95%以上的帧被成功跟踪
        fail, outlst = getlst(bboxlist, ifsquare)         #尝试随机抽取20对帧
        if not fail:                                      #如果抽取成功
            for i in range(20):
                nm1 = lst[k][0:len(lst[k])-4] + '_' + str(i+1) + '_1.jpg'            #远景照片文件名
                nm2 = lst[k][0:len(lst[k])-4] + '_' + str(i+1) + '_2.jpg'            #远景照片文件名
                imm1 = cache1[outlst[i][0]].copy()
                imm2 = cache1[outlst[i][2]].copy()
                bbox1 = outlst[i][1]
                bbox2 = outlst[i][3]
                im1 = cv2.cvtColor(imm1, cv2.COLOR_BGR2RGB)
                im2 = cv2.cvtColor(imm2, cv2.COLOR_BGR2RGB)
                a1 = Image.fromarray(np.uint8(im1))
                a2 = Image.fromarray(np.uint8(im2))
                a1.save(os.path.join(path1, nm1))
                a2.save(os.path.join(path1, nm2))
                
                lx1 = int(bbox1[0])
                tx1 = int(bbox1[1])
                w1 = int(bbox1[2]) - lx1
                h1 = int(bbox1[3]) - tx1
                cv2.rectangle(imm1,(lx1,tx1),(lx1 + w1,tx1 + h1),(0,255,0),7)
                lx2 = int(bbox2[0])
                tx2 = int(bbox2[1])
                w2 = int(bbox2[2]) - lx2
                h2 = int(bbox2[3]) - tx2
                cv2.rectangle(imm2,(lx2,tx2),(lx2 + w2,tx2 + h2),(0,255,0),7)
                im1 = cv2.cvtColor(imm1, cv2.COLOR_BGR2RGB)
                im2 = cv2.cvtColor(imm2, cv2.COLOR_BGR2RGB)
                a1 = Image.fromarray(np.uint8(im1))
                a2 = Image.fromarray(np.uint8(im2))
                a1.save(os.path.join(path2, nm1))
                a2.save(os.path.join(path2, nm2))
                f.write(', '.join(['0', '1', str(size[0]), str(size[1]), str(lx1), str(tx1), str(w1), str(h1), os.path.join(path1, nm1)]) + '\n')
                f.write(', '.join(['0', '1', str(size[0]), str(size[1]), str(lx2), str(tx2), str(w2), str(h2), os.path.join(path1, nm2)]) + '\n')
                
            srcv = open(video_full_path, "rb") 
            dstv = open(os.path.join(path3, video_name), "wb") 
            dstv.write(srcv.read())
            srcv.close() 
            dstv.close() 
            
            print(video_name, '成功！')
    else:                                                  #如果没有95%以上的帧跟踪成功
        fail = True
    if fail:                                               #如果成功跟踪帧数少于95%或未抽取到足够多的帧
        cache = cache[int(len(cache)*0.3):len(cache)]      #去掉前30%的帧重新跟踪
        bboxb = backprop(cache)
        bboxf = frontprop(cache)
        bboxlist = combinebbox(bboxb, bboxf, shrink_rate)
        snum = 0
        fail = False
        fNUMS = len(bboxlist)
        for i in range(fNUMS):
            if i == 0 or i == fNUMS - 1:
                bbox = bboxlist[i]
            else:
                bbox = getbbox(i,bboxlist)
            if not bbox == (0,0,0,0):
                snum += 1
        if snum/fNUMS > 0.95:
            fail, outlst = getlst(bboxlist, ifsquare)
            if not fail:
                for i in range(20):
                    nm1 = lst[k][0:len(lst[k])-4] + '_' + str(i+1) + '_1.jpg'             #近景照片文件名
                    nm2 = lst[k][0:len(lst[k])-4] + '_' + str(i+1) + '_2.jpg'             #远景照片文件名
                    imm1 = cache1[outlst[i][0]].copy()
                    imm2 = cache1[outlst[i][2]].copy()
                    bbox1 = outlst[i][1]
                    bbox2 = outlst[i][3]
                    im1 = cv2.cvtColor(imm1, cv2.COLOR_BGR2RGB)
                    im2 = cv2.cvtColor(imm2, cv2.COLOR_BGR2RGB)
                    a1 = Image.fromarray(np.uint8(im1))
                    a2 = Image.fromarray(np.uint8(im2))
                    a1.save(os.path.join(path1, nm1))
                    a2.save(os.path.join(path1, nm2))

                    lx1 = int(bbox1[0])
                    tx1 = int(bbox1[1])
                    w1 = int(bbox1[2]) - lx1
                    h1 = int(bbox1[3]) - tx1
                    cv2.rectangle(imm1,(lx1,tx1),(lx1 + w1,tx1 + h1),(0,255,0),7)
                    lx2 = int(bbox2[0])
                    tx2 = int(bbox2[1])
                    w2 = int(bbox2[2]) - lx2
                    h2 = int(bbox2[3]) - tx2
                    cv2.rectangle(imm2,(lx2,tx2),(lx2 + w2,tx2 + h2),(0,255,0),7)
                    im1 = cv2.cvtColor(imm1, cv2.COLOR_BGR2RGB)
                    im2 = cv2.cvtColor(imm2, cv2.COLOR_BGR2RGB)
                    a1 = Image.fromarray(np.uint8(im1))
                    a2 = Image.fromarray(np.uint8(im2))
                    a1.save(os.path.join(path2, nm1))
                    a2.save(os.path.join(path2, nm2))
                    f.write(', '.join(['0', '1', str(size[0]), str(size[1]), str(lx1), str(tx1), str(w1), str(h1), os.path.join(path1, nm1)]) + '\n')
                    f.write(', '.join(['0', '1', str(size[0]), str(size[1]), str(lx2), str(tx2), str(w2), str(h2), os.path.join(path2, nm2)]) + '\n')

                srcv = open(video_full_path, "rb") 
                dstv = open(os.path.join(path3, video_name), "wb") 
                dstv.write(srcv.read())
                srcv.close() 
                dstv.close() 
                print(video_name, '后半成功！')
        else:
            fail = True
    if fail:
        srcv = open(video_full_path, "rb") 
        dstv = open(os.path.join(path5, video_name), "wb") 
        dstv.write(srcv.read())
        srcv.close() 
        dstv.close() 
        
        fail, outlst = getlst2(fNUMS)
        if not fail:
            for i in range(20):
                nm1 = lst[k][0:len(lst[k])-4] + '_' + str(i+1) + '_1.jpg'            #近景照片文件名
                nm2 = lst[k][0:len(lst[k])-4] + '_' + str(i+1) + '_2.jpg'            #远景照片文件名
                imm1 = cache1[outlst[i][0]].copy()
                imm2 = cache1[outlst[i][1]].copy()
                im1 = cv2.cvtColor(imm1, cv2.COLOR_BGR2RGB)
                im2 = cv2.cvtColor(imm2, cv2.COLOR_BGR2RGB)
                a1 = Image.fromarray(np.uint8(im1))
                a2 = Image.fromarray(np.uint8(im2))
                a1.save(os.path.join(path4, nm1))
                a2.save(os.path.join(path4, nm2))

        print(video_name, '失败！')
f.close()
