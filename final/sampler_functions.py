import cv2
import numpy as np
from os.path import join
import os
from collections import defaultdict
import pickle
from shutil import copy
import math
from sklearn.cluster import KMeans
from sklearn import svm

class SR:

    def get_category(self, base_dir):
        sub_dir = [join(base_dir, sub) for sub in os.listdir(base_dir)]
        return sub_dir

    def split_data(self, base_dir, train_dir, test_dir):
        for dir in base_dir:
            videos = os.listdir(dir)
            print("splitting data under "+dir)
            for i in range(len(videos)):
                if ".avi" in videos[i]:
                    if(i%2 == 0):
                        copy(dir+"/"+videos[i],train_dir)
                    else:
                        copy(dir+"/"+videos[i],test_dir)
        return


    def create_key_frame(self, video_dir,frame_dir):
        for file in os.listdir(video_dir):
            f = join(video_dir,file)
            cap = cv2.VideoCapture(f)
            fps = cap.get(cv2.CAP_PROP_FPS)
            f_n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            len =  int(f_n/fps)
            #print(f, fps, f_n, len)
            for i in range(4):
                cap.set(cv2.CAP_PROP_POS_MSEC,len*1000*i/4)
                ret, frame = cap.read()
                key_frame_name = frame_dir + "/" + file[:-4]+"_"+str(i)+".png"
                print(key_frame_name)
                if ret:
                    cv2.imwrite(key_frame_name, frame)
            cap.release()
        return

    def frame_category(self,frame_dir):
        label = defaultdict(list)
        for f in os.listdir(frame_dir):
            f_l = f[2:-14]
            label[f_l].append(frame_dir+"/"+f)
        #print(label,len(label))
        return label

    def cal_dof(self,frame_dir, dof_dir):
        frames = os.listdir(frame_dir)
        for i in range(len(frames)-1):
            #print(frames[i])
            if(i%4 !=3):
                if(i%4 == 0):
                    print(i)
                    fsplit = frames[i].split('/')
                    fname = dof_dir+"/"+fsplit[-1][:-4]+"_flow.txt"
                    fopen = open(fname,'wb')
                    print('Adding the dof in file: '+ fname)
                #print(dir+'/'+frames[i])
                frame1 = cv2.imread(frame_dir+'/'+frames[i])
                frame2 = cv2.imread(frame_dir+'/'+frames[i+1])
                pvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                #hsv = np.zeros_like(frame1)
                #hsv[...,1]=255
                flow = cv2.calcOpticalFlowFarneback(pvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                pickle.dump(flow,fopen)
            else:
                fopen.close()

            #dof[category]=flow_list
        return

    # def cal_hog(self,label):
    #     winSize = (64,64)
    #     blockSize = (16,16)
    #     blockStride = (8,8)
    #     cellSize = (8,8)
    #     nbins = 9
    #     derivAperture = 1
    #     winSigma = 4.
    #     histogramNormType = 0
    #     L2HysThreshold = 2.0000000000000001e-01
    #     gammaCorrection = 0
    #     nlevels = 8
    #     hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #     winStride = (16,16)
    #     padding = (8,8)
    #     locations = ((10,10),)
    #     fname = ''
    #     for category,frames in label.items():
    #         print('computing HOG in category: '+ category)
    #         for i in range(len(frames)):
    #             frame = cv2.imread(frames[i])
    #             hist = hog.compute(frame,winStride,padding,locations)
    #             if(i%4 == 0):
    #                 fsplit = frames[i].split('/')
    #                 fname = "/Users/qiuwshou/Desktop/hog/"+fsplit[-1][:-4]+"_hog.txt"
    #                 print('computing the HOG in file: '+ fname)
    #                 fopen = open(fname,'wb')
    #                 pickle.dump(hist,fopen)
    #             else:
    #                 hist = hog.compute(frame,winStride,padding,locations)
    #                 pickle.dump(hist,fopen)
    #                 if(i%4 == 3):
    #                     fopen.close()
    #                     fname = ''
    #     return

    def codebook_dof(self,dof_dir,output_dir, num_cent,threshold):

        s_x, s_y = [], []
        for f in os.listdir(dof_dir):
            m_x, m_y = [], []
            print("adding dof in "+f)
            file = open(join(dof_dir,f),'rb')
            for i in range(3):
                dof = pickle.load(file)
                for height in range(dof.shape[0]):
                    for width in range(dof.shape[1]):
                        v = dof[height,width]
                        disp = math.sqrt(v[0]**2+v[1]**2)
                        #print(v,disp)
                        if(disp>threshold):
                            m_x.append(v[0])
                            m_y.append(v[1])
                        # else:
                        #     s_x.append(v[0])
                        #     s_y.append(v[1])
            file.close()
            m_z = np.vstack((np.asarray(m_x),np.asarray(m_y)))
        # s_z = np.vstack((np.asarray(s_x),np.asarray(s_y)))
            m_z = np.transpose(m_z)
            m_z = np.float32(m_z)
            # fopen = open(output_dir+"all_feature.txt",'wb')
            # pickle.dump(m_z,fopen)
            # fopen.close()
        # s_z = np.transpose(s_z)
        # s_z = np.float32(s_z)
        # define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        #ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            ret,label,center=cv2.kmeans(m_z,num_cent,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            fopen = open(output_dir+f[:-4]+"_hist.txt",'wb')
            pickle.dump(center,fopen)
            fopen.close()

        # ret,label,center=cv2.kmeans(s_z,num_cent,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # fopen = open(output_dir+"stationary_ck.txt",'wb')
        # pickle.dump(center,fopen)
        # fopen.close()
        return

    def codebook_dof_all(self,dir, output,num_cent):
        x,y = [],[]
        for f in os.listdir(dir):
            fopen = open(dir+"/"+f,'rb')
            print("processing: "+dir+"/"+f)
            cent = pickle.load(fopen)
            for i in range(cent.shape[0]):
                x.append(cent[i][0])
                y.append(cent[i][1])
        z = np.vstack((np.asarray(x),np.asarray(y)))
        z = np.transpose(z)
        z = np.float32(z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        ret,label,center=cv2.kmeans(z,num_cent,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        fopen = open(output,'wb')
        pickle.dump(center,fopen)
        fopen.close()
        return

    def hist_dof(self,cent_dir,dof_dir,output,label_dir):
        x,y = [],[]
        label = []
        fopen = open(cent_dir,'rb')
        cent = pickle.load(fopen)
        fopen.close()
        frames = os.listdir(dof_dir)

        fhist = open(output,'wb')
        for frame in frames:
            category = frame[2:-24]
            if(category in label):
                category = label.index(category)
            else:
                label.append(category)
                category = label.index(category)
            hist = [0]*200
            fopen = open(dof_dir+"/"+frame, 'rb')
            print("processing:" + dof_dir+"/"+frame)
            dof = pickle.load(fopen)
            fopen.close()
            for i in range(len(dof)):
                    v = dof[i]
                    disp = math.sqrt(v[0]**2+v[1]**2)
                    if(disp>2):
                        idx = self.find_nearest(np.asarray(v),np.asarray(cent))
                            #print(idx,len(cent),cent)
                        hist[idx] += 1
            x.append(hist)
            #print(category)
            y.append(category)
        pickle.dump(x,fhist)
        pickle.dump(y,fhist)
        fhist.close()
        # fopen = open(label_dir,'wb')
        # pickle.dump(label,fopen)
        # fopen.close()
        return


    def find_nearest(self,value, array):
        idx = 999999
        min = 999999
        for i in range(len(array)):
            sum = 0
            for j in range(len(value)):
                sum += (value[j]-array[i][j])**2
            sum = math.sqrt(sum)
            if(sum<min):
                min = sum
                idx = i
        return idx

    def codebook_csift(self,csift_dir,dof_dir,output_dir,num_cent,t):
        stationary, moving = [],[]
        frames = os.listdir(csift_dir)
        for i in range(len(frames)-1):
            fname = frames[i]
            print("processing:"+ csift_dir+"/"+fname)
            if(i%4 != 3):
                if(i%4 ==0):
                    fdof = open(dof_dir+"/"+fname[:-4]+"_flow.txt","rb")
                dof = pickle.load(fdof)
                fcsift = open(csift_dir+"/"+fname,'r')
                content = fcsift.readlines()
                index,value = self.get_index_value(content)

                for k in range(len(index)):
                    height,width = int(index[k][1]),int(index[k][0])
                    desc = [int(j) for j in value[k]]
                    v = dof[height][width]
                    if(self.cal_motion(v)>t):
                        moving.append(desc)
                    else:
                        stationary.append(desc)
                fcsift.close()
            else:
                fdof.close()
        stationary=np.asarray(stationary)
        moving=np.asarray(moving)

        fstaionary= open(output_dir+"/"+"all_sp.txt",'wb')
        fmoving = open(output_dir+"/"+"all_mp.txt",'wb')
        pickle.dump(stationary,fstaionary)
        pickle.dump(moving,fmoving)
        fstaionary.close()
        fmoving.close()

        print("calculating kmean......")
        kmeans_sp = KMeans(n_clusters=num_cent, random_state=0).fit(stationary)
        kmeans_mp = KMeans(n_clusters=num_cent, random_state=0).fit(moving)

        fsp = open(output_dir+"/"+"sp_cent.txt",'wb')
        fmp = open(output_dir+"/"+"mp_cent.txt",'wb')
        pickle.dump(kmeans_mp,fmp)
        pickle.dump(kmeans_sp,fsp)
        fsp.close()
        fmp.close()
        return

    def hist_csift(self,cent_dir,csift_dir,dof_dir,output_dir,num_cent,t):
        x,y = [],[]
        label = []
        fopen1 = open(cent_dir+"/"+"sp_cent.txt",'rb')
        fopen2 = open(cent_dir+"/"+"mp_cent.txt",'rb')
        cent_sp  = pickle.load(fopen1)
        cent_mp = pickle.load(fopen2)
        fopen1.close()
        fopen2.close()
        frames = os.listdir(csift_dir)

        fhist = open(output_dir+"/"+"hist_all.txt",'wb')

        for i in range(len(frames)-1):
            hist_sp,hist_mp = [0]*num_cent,[0]*num_cent
            fname = frames[i]

            category = fname[2:-14]
            if(category in label):
                category = label.index(category)
            else:
                label.append(category)
                category = label.index(category)

            print("processing:"+ csift_dir+"/"+fname)

            if(i%4 != 3):
                if(i%4 ==0):
                    fdof = open(dof_dir+"/"+fname[:-4]+"_flow.txt","rb")
                dof = pickle.load(fdof)
                fcsift = open(csift_dir+"/"+fname,'r')
                content = fcsift.readlines()
                index,value = self.get_index_value(content)

                for k in range(len(index)):
                    height,width = int(index[k][1]),int(index[k][0])
                    desc = [int(j) for j in value[k]]
                    v = dof[height][width]
                    if(self.cal_motion(v)>t):
                        idx = self.find_nearest(desc,cent_mp)
                        hist_mp[idx] += 1
                    else:
                        idx = self.find_nearest(desc,cent_sp)
                        hist_sp[idx] += 1

                x.append(hist_sp+hist_mp)
                y.append(category)
                fcsift.close()
            else:
                fdof.close()

        pickle.dump(x,fhist)
        pickle.dump(y,fhist)


        fhist.close()
        # fopen = open(output_dir+"/"+"csift_label.txt",'wb')
        # pickle.dump(label,fopen)
        # fopen.close()
        return


    def get_index_value(self,content):
        content = content[3:]
        index = [i.split(";")[0:2][0].split(" ")[1:3] for i in content]
        desc = [i.split(";")[0:2][1].split(" ")[1:] for i in content]
        return index,desc


    def cal_motion(self,v):
        disp = math.sqrt(v[0]**2+v[1]**2)
        return disp


    def svm_cls(self,train,test):
        ft = open(train,'rb')
        x = pickle.load(ft)
        y = pickle.load(ft)
        ft.close()

        fp = open(test,'rb')
        predict_x = pickle.load(fp)
        fp.close()


        lin_clf = svm.LinearSVC(C=1.0,penalty='l2',loss='squared_hinge',dual=False,tol=1e-4,multiclass='ovr',fit_intercept=True
                                ,intercept_scaling=1,class_weight=None,verbose=0,random_state=None,max_iter=100)
        lin_clf.fit(x,y)
        parameter = lin_clf.get_params(deep=True)
        predict_y = lin_clf.predict(predict_x)
        print(predict_y,parameter)
        return




