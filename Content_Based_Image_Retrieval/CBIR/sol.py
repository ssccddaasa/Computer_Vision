import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import skew, mode, kurtosis
from scipy.spatial import distance
from skimage.feature import hog
import time



def imgeload(imgnum):
    img = cv2.imread("Images/"+str(imgnum)+".jpg")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img2

def color_hist(img,b):
    fet = []
    for i in range(3):
        histo = cv2.calcHist([img], [i], None, [b], [0, 256])
        nhisto = histo / histo.sum()
        fet.append(nhisto)

    feat_vec = np.concatenate((fet[0].flatten(),fet[1].flatten(),fet[2].flatten()))
    return feat_vec

def color_mom(img):
    mean = np.mean(img)
    std = np.std(img)
    skewness = skew(img.flatten())
    feat_vec = np.array([mean,std,skewness])
    return feat_vec

def color_mom2(img):
    mean = np.mean(img)
    std = np.std(img)
    skewness = skew(img.flatten())
    med = np.median(img)
    mode_v = mode(img.reshape(-1,3),axis=None)
    kur = kurtosis(img.flatten())
    feat_vec = np.array([mean,std,skewness,med,mode_v[0],mode_v[1],kur])
    return feat_vec

def color_hist_HOG(img):
    fet = []
    for i in range(3):
        histo = cv2.calcHist([img], [i], None, [120], [0, 256])
        nhisto = histo / histo.sum()
        fet.append(nhisto)

    his_vec = np.concatenate((fet[0].flatten(),fet[1].flatten(),fet[2].flatten()))    
    imggry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_fet = hog(imggry, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), block_norm='L2-Hys')
    nhis_vec = his_vec / np.linalg.norm(his_vec)
    nhog_fet = hog_fet / np.linalg.norm(hog_fet)
    feat_vec = np.concatenate((nhis_vec,nhog_fet))
    return feat_vec




def His_to_str(imgnum,vec):
    strres = ""
    for i in range(len(vec)-1):
        strres += str(vec[i]) + ","

    strres += str(vec[-1]) + "_" + str(imgnum) +"\n" 

    return strres

def train(fi,op,b):
    start_time = time.time()
    f = open(fi,"w")
    for i in range(1000):
        img = imgeload(i)
        if(op == "1"):
            feat_vec = color_hist(img,b)
        elif(op == "2"):
            feat_vec = color_mom(img)
        elif(op == "3"):
            feat_vec = color_mom2(img)
        else:
            feat_vec = color_hist_HOG(img)
        s = His_to_str(i,feat_vec)
        f.write(s)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")    

def CBIR(img_name,fi,op,b,worn):
    w = np.array([1,0.5,10])
    w2 = np.array([1,0.5,10,1,2,2,3])
    img = cv2.imread(img_name)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if(op == "1"):
        vec_tst = color_hist(img2,b)
    elif(op == "2"):
        vec_tst = color_mom(img2)
    elif(op == "3"):
        vec_tst = color_mom2(img2)
    else:
        vec_tst = color_hist_HOG(img2)
    fr = open(fi,"r")
    dis_res = []
    for x in fr:
        x_split = x.split("_")
        nums = x_split[0].split(",")
        numsnp = np.array(nums)
        vec = numsnp.astype(np.float32)
        if(worn == 1):
            if(op == "2"):
                E_D =  distance.euclidean(vec_tst*w, vec*w)
            elif(op == "3"):
                E_D =  distance.euclidean(vec_tst*w2, vec*w2)
        else:
            E_D =  distance.euclidean(vec_tst, vec)
        dis_res.append(E_D)

    dis_resnp = np.array(dis_res)
    return dis_resnp




def truep (numimg):

    hun = numimg // 100
    res = np.arange(hun*100, (hun+1)*100)
    return res

def metrics(dist,thre,numimg):
    re_image = np.where(dist <= thre)[0]
    all_rele = len(re_image)
    pos = truep(numimg)

    true_pos = len(set(re_image).intersection(pos))
    fp = all_rele - true_pos
    tn = len(dist) - all_rele - (len(pos)-true_pos)
    acc = (true_pos + tn) / len(dist)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = true_pos / all_rele if all_rele > 0 else 0
    recall = true_pos / len(pos) if len(pos) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision ,recall, f1_score, fpr, acc

def ROC(qur,file,op,b,worn):
    start_time = time.time()

    psa = [0] * 100
    rsa = [0] * 100
    fsa = []
    fpr = [0] * 100
    Acc = [0] * 100
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    AUC_VAL = []
    for q, ax in zip(quries,axs.flat):
        dis = CBIR("test/"+str(q)+".jpg",file,op,b,worn)

        PS = []
        RS = []
        fpl = []
        ac = []
        thr_val = np.linspace(min(dis),max(dis), num=100)
        for thre in thr_val:
            p,r,f,fp,a = metrics(dis,thre,q)
            PS.append(p)
            RS.append(r)
            fsa.append(f)
            fpl.append(fp)
            ac.append(a)
        
        psa[:] = [x + y for x, y in zip(psa, PS)]
        rsa[:] = [x + y for x, y in zip(rsa, RS)]
        fpr[:] = [x + y for x, y in zip(fpr, fpl)]
        Acc[:] = [x + y for x, y in zip(Acc, ac)]

        auc = np.trapz(RS, x=fpl)
        AUC_VAL.append(auc)

        ax.plot(fpl, RS, marker='o')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(f'Subplot ROC {q}')
        ax.grid(True)
    

    Acc[:] = [x/len(qur) for x in Acc]
    AcC = sum(Acc) / len(Acc)
    #print("AVG Accuracy: ", AcC)
    AUC_AVG = np.mean(AUC_VAL)
    print("Average AUC:", AUC_AVG)
    psa[:] = [x/len(qur) for x in psa]
    pr = sum(psa) / len(psa)
    rsa[:] = [x/len(qur) for x in rsa]
    re = sum(rsa) / len(rsa)
    F1 = sum(fsa) / len(fsa)
    print("AVG F1: ", F1)   
    print("AVG Precision: ", pr)
    print("AVG Recall: ", re)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    plt.tight_layout()  # Adjust layout for better appearance
    plt.show()
    plt.plot(fpr, rsa, marker='o')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (AVG)')
    plt.grid(True)
    plt.show()

def Show_Img(dis_resnp):
    ret_inx = np.argpartition(dis_resnp, 10)[:10]
    sor_val = dis_resnp[ret_inx]
    sor_ind = np.argsort(sor_val)  # Sort indices based on values
    sor_ret_inx = ret_inx[sor_ind]
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

    #Loop through image files and display each image
    j = 1
    for i, ax in enumerate(axes.flat):
        img = mpimg.imread("Images/"+str(sor_ret_inx[i])+".jpg")
        ax.imshow(img)
        ax.axis('off')  # Hide axis labels
        ax.set_title('Image_num: '+ str(j))
        j += 1

# Show the plot with ten images
    plt.tight_layout()
    plt.show()


quries = [16,135,221,319,465,585,649,738,871,991]


print("Welcome to my CBIR\n")

while(1):
    print("what do you want: 1-) train or 2-) CBIR or 3-) test")
    print("end for END")
    ch = input("put your choice: ")

    if(ch == "1"):

        print("choose the algorithm: ")
        print("1-) color histogram")
        print("2-) color moments1")
        print("3-) color moments2")
        print("4-) color histogram + HOG type")
        op = input("put your choice: ")

        if(op == "1"):
            print("how many bins do you want?")
            print("a. 120")
            print("b. 180")
            print("c. 360")
            op2 = input("put your choice: ")
            if(op2 == "a"):
                train("databaseCH120.txt","1",120)
                print("train is done")
            elif(op2 == "b"):
                train("databaseCH180.txt","1",180)
                print("train is done")
            elif(op2 == "c"):
                train("databaseCH360.txt","1",360)
                print("train is done")
            else:
                print("invalid choice")
        
        
        elif(op == "2"):
            train("databaseCM1.txt","2",120)
            print("train is done")
        
        elif(op == "3"):
            train("databaseCM2.txt","3",120)
            print("train is done")
        
        elif(op == "4"):
            train("databaseCHHOG.txt","4",120)
            print("train is done")
        else:
            print("invalid choice")
    
    elif(ch == "2"):
      
        print("choose the algorithm: ")
        print("1-) color histogram 120 pin")
        print("2-) color histogram 180 pin")
        print("3-) color histogram 360 pin")
        print("4-) color moments1 wetighed")
        print("5-) color moments1 not")
        print("6-) color moments2 wetighed")
        print("7-) color moments2 not")
        print("8-) color histogram + HOG type")
        op = input("put your choice: ")
        opim = input("put the number of the test image (1,2,3): ")
        if(opim == "1"):
            img_name = "test.jpg"
        elif(opim == "3"):
            img_name = "test3.jpg"
        else:
            img_name = "test2.jpg"
        if( op == "1"):
            dis = CBIR(img_name,"databaseCH120.txt","1",120,0)
            Show_Img(dis)
        elif( op == "2"):
            dis = CBIR(img_name,"databaseCH180.txt","1",180,0)
            Show_Img(dis)
        elif( op == "3"):
            dis = CBIR(img_name,"databaseCH360.txt","1",360,0)
            Show_Img(dis)
        elif( op == "4"):
            dis = CBIR(img_name,"databaseCM1.txt","2",120,1)
            Show_Img(dis)
        elif( op == "5"):
            dis = CBIR(img_name,"databaseCM1.txt","2",120,0)
            Show_Img(dis)
        elif( op == "6"):
            dis = CBIR(img_name,"databaseCM2.txt","3",120,1)
            Show_Img(dis)
        elif( op == "7"):
            dis = CBIR(img_name,"databaseCM2.txt","3",120,0)
            Show_Img(dis)
        elif( op == "8"):
            dis = CBIR(img_name,"databaseCHHOG.txt","4",120,0)
            Show_Img(dis)
        else:
            print("invalid choice")
    

    elif(ch == "3"):
        print("choose the algorithm: ")
        print("1-) color histogram 120 pin")
        print("2-) color histogram 180 pin")
        print("3-) color histogram 360 pin")
        print("4-) color moments1 wetighed")
        print("5-) color moments1 not")
        print("6-) color moments2 wetighed")
        print("7-) color moments2 not")
        print("8-) color histogram + HOG type")
        op = input("put your choice: ")

        if( op == "1"):
            ROC(quries,"databaseCH120.txt","1",120,0)
        elif( op == "2"):
            ROC(quries,"databaseCH180.txt","1",180,0)
        elif( op == "3"):
            ROC(quries,"databaseCH360.txt","1",360,0)
        elif( op == "4"):
            ROC(quries,"databaseCM1.txt","2",120,1)
        elif( op == "5"):
            ROC(quries,"databaseCM1.txt","2",120,0)
        elif( op == "6"):
            ROC(quries,"databaseCM2.txt","3",120,1)
        elif( op == "7"):
            ROC(quries,"databaseCM2.txt","3",120,0)
        elif( op == "8"):
            ROC(quries,"databaseCHHOG.txt","4",120,0)
        else:
            print("invalid choice")


    elif(ch == "end"):
        break
    else:
        print("invalid choice")







