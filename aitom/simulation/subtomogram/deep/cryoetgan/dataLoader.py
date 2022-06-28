import numpy as np
import pickle


def load_picdata(imgsize, den_path, sub_path, X_noise=0.0, label=4):
    image_size = imgsize
    np.random.seed(0)
    with open(den_path, 'rb') as den:
      pic_den = pickle.load(den,  encoding='latin1')

    with open(sub_path, "rb") as pic:
      pic_sub = pickle.load(pic,  encoding='latin1')#,  encoding='latin1')

    img_max_num = {}
    density = None
    den_label = None
    if label == 7:
      id2label = {"31": 0, "33": 1, "35": 2, "43": 3, "69": 4, "72": 5, "73": 6}
    else:
      id2label = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s":3}
    
    if label == 4:
      subtomogram = None
      density_name_order = []
      
      for idx, dens in pic_den.items():
        density_name_order.append(idx)
        maxnum = 400
        if density is None:
          if len(dens)>maxnum:
            density = np.array(dens[0:maxnum])
            den_label = np.ones(maxnum)*id2label[idx]
          else:
            dens = np.array(dens)
            tmpden = dens
            for exi in range(int(400/len(dens))):
              tmpden = np.concatenate((tmpden, np.array(dens[0:min(dens.shape[0], 400-dens.shape[0]*(1+exi))])), axis=0)
            density = np.array(tmpden[0:maxnum])
            den_label = np.ones(maxnum)*id2label[idx]
        else:
          if len(dens) > maxnum:
            density = np.concatenate((density, np.array(dens[0:maxnum])), axis=0)
            den_label = np.concatenate((den_label, np.ones(maxnum)*id2label[idx]), axis=0)
          else:
            dens = np.array(dens)
            tmpden = dens
            for exi in range(int(400/len(dens))):
              tmpden = np.concatenate((tmpden, np.array(dens[0:min(dens.shape[0], 400-dens.shape[0]*(1+exi))])), axis=0)
            density = np.concatenate((density, tmpden), axis=0)
            den_label = np.concatenate((den_label, np.ones(maxnum)*id2label[idx]), axis=0)

      for idx in density_name_order:
      # for idx, subs in pic_sub.items():
        subs = pic_sub[idx]
        img_max_num[idx] = 400
        if subtomogram is None:
          if len(subs)>400:
            subtomogram = np.array(subs[0:min(400, len(subs))])
          else:
            subs = np.array(subs)
            tmpsub = subs
            for exi in range(int(400/len(subs))):
              tmpsub = np.concatenate((tmpsub, np.array(subs[0:min(subs.shape[0], 400-subs.shape[0]*(1+exi))])), axis=0)
            print(tmpsub.shape[0])
            subtomogram = np.array(tmpsub)
        else:
          if len(subs) > 400:
            subtomogram = np.concatenate((subtomogram, np.array(subs[0:min(400, len(subs))])), axis=0)
          else:
            subs = np.array(subs)
            tmpsub = subs
            for exi in range(int(400/len(subs))):
              tmpsub = np.concatenate((tmpsub, np.array(subs[0:min(subs.shape[0], 400-subs.shape[0]*(1+exi))])), axis=0)
            print(tmpsub.shape[0])
            subtomogram = np.concatenate((subtomogram, tmpsub), axis=0)

    else:

      for idx, dens in pic_den.items():
        img_max_num[idx] = dens.shape[0]
        if density is None:
          density = np.array(dens)
          den_label = np.ones(density.shape[0])*id2label[idx]
        else:
          density = np.concatenate((density, np.array(dens)), axis=0)
          den_label = np.concatenate(
              (den_label, np.ones(dens.shape[0])*id2label[idx]), axis=0)

      for idx, num in img_max_num.items():
        sub = np.array(pic_sub[idx])
        pic_sub[idx] = sub
        if sub.shape[0] < num:
          # expand the number of subtomograms
          for exi in range(int(num/sub.shape[0])):
            pic_sub[idx] = np.concatenate((pic_sub[idx], sub[0:min(sub.shape[0], num-sub.shape[0]*(1+exi))]), axis=0)
          print(pic_sub[idx].shape[0])
        else:
          pic_sub[idx] = sub[0:num, :]
          print(num)

        print(pic_sub[idx].shape)

      subtomogram = None
      for idx, subs in pic_sub.items():
        if subtomogram is None:
          subtomogram = subs
        else:
          subtomogram = np.concatenate((subtomogram, subs), axis=0)

      # Guass Noies
    if not X_noise == 0.0:
      print("adding noise, sigma is {}".format(X_noise))
      noise = np.random.normal(0, X_noise, size=density.shape)
      density = density + noise
    #print("add noise density {}".format(density.shape))

    #pic_den = np.load('./data/'+FLAGS.X+'.npy')

    print("density shape: ", density.shape)
    print("subtomogram shape: {}".format(subtomogram.shape))
    print("density label: {}".format(den_label.shape))

    total_den_num = density.shape[0]
    total_sub_num = subtomogram.shape[0]
    max_imgnum = min(total_den_num, total_sub_num)

    img_den = density

    img_den = preprocessz(img_den)

    img_sub = subtomogram

    img_sub = preprocessz(img_sub)

    return img_den,img_sub,den_label,max_imgnum


def preprocessz(X_all):
  X_all = X_all - np.mean(X_all)
  X_all = X_all / np.std(X_all)
  return X_all


def Toslice(imgsize, img_den, img_sub, den_label):
    sub_slice = []
    for i in range(img_sub.shape[0]):
        sub = img_sub[i,:].reshape((imgsize,imgsize,imgsize))
        for img_i in range(imgsize):
            sub_slice.append(sub[:,:,img_i].reshape((imgsize,imgsize)))
    print("subtomograms slice {}".format(len(sub_slice)))

    den_slice = []
    label_slice = []
    for i in range(img_den.shape[0]):
        den = img_den[i,:].reshape((imgsize,imgsize,imgsize))
        for img_i in range(imgsize):
            den_slice.append(den[:,:,img_i].reshape((imgsize,imgsize)))
            label_slice.append(den_label[i])

    print("density slice {}".format(len(label_slice)))
    return np.array(den_slice), np.array(sub_slice), np.array(label_slice)
  

def read_data(pic_sub_dir, imgsize, imgnum=None):
    with open(pic_sub_dir, "rb") as pic:
        pic_sub = pickle.load(pic,  encoding='latin1')
    for idx, img in pic_sub.items():
        if (imgnum is None):
            pic_sub[idx] = np.array(pic_sub[idx]).reshape((-1, imgsize, imgsize, imgsize,1))
        else:
          if (len(img) >= imgnum):
              pic_sub[idx] = np.array(pic_sub[idx]).reshape((-1, imgsize, imgsize, imgsize,1))
          else:
              img = np.array(img)
              pic_sub[idx] = np.array(pic_sub[idx]).reshape((-1, imgsize, imgsize, imgsize,1))
              for exi in range(int(imgnum/img.shape[0])):
                  pic_sub[idx] = np.concatenate((pic_sub[idx], img[0:min(img.shape[0], imgnum-img.shape[0]*(1+exi))]), axis=0)

    return pic_sub


def get_picdata(data_path='./data/Noble_7_classes.pickle', imgsize = 40,label = 4):
    # new_sub.pickle  same_density_wgan_sub.pickle
    pic_sub = read_data(data_path, imgsize)

    if label == 7:
      id2label = {"31": 0, "33": 1, "35": 2, "43": 3, "69": 4, "72": 5, "73": 6}
    else:
      id2label = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s":3}
    x_train = None
    xval = None
    y_train = None
    y_val = None

    for idx, img in pic_sub.items():
        num = img.shape[0]
        np.random.shuffle(img)
        if not imgsize == 28:
          if num<400:
            trainX = img.reshape((-1, imgsize, imgsize, imgsize, 1))[0:100,:]
            valX = img.reshape((-1, imgsize, imgsize, imgsize, 1))[100:, :]
            #trainX = preprocessz(trainX)
            #valX = preprocessz(valX)
          else:
            trainX = img[0:400,:].reshape((-1, imgsize, imgsize, imgsize, 1))
            valX = img[400:, :].reshape((-1, imgsize, imgsize, imgsize, 1))
            #trainX = preprocessz(trainX)
            #valX = preprocessz(valX)
        else:
          trainX = img[0:int(num*validation_split),:].reshape((-1, imgsize, imgsize, imgsize, 1))
          valX = img[int(num*validation_split):, :].reshape((-1, imgsize, imgsize, imgsize, 1))
          #trainX = preprocessz(trainX)
          #valX = preprocessz(valX)
        if x_train is None:
            x_train = trainX
            x_val = valX
            y_train = np.ones(trainX.shape[0])*id2label[idx]
            if not valX is None:
              y_val = np.ones(valX.shape[0])*id2label[idx]

        else:
            x_train = np.concatenate((x_train, trainX), axis=0)
            
           
            y_train = np.concatenate((y_train, np.ones(trainX.shape[0])*id2label[idx]), axis=0)
            
            if not x_val is None:
              if not valX is None:
                x_val = np.concatenate((x_val, valX), axis=0)
                y_val = np.concatenate((y_val, np.ones(valX.shape[0])*id2label[idx]), axis=0)
            else:
              if not valX is None:
                x_val = valX
                y_val = np.ones(valX.shape[0])*id2label[idx]
            
    x_train = preprocessz(x_train)
    x_val = preprocessz(x_val)
    np.save("./data-old/train_data", x_train)
    np.save("./data-old/test_data", x_val)
    np.save("./data-old/train_label", y_train)
    np.save("./data-old/test_label", y_val)

    print("x_train ", x_train.shape)
    print("x_val ", x_val.shape)
    train_index = np.arange(x_train.shape[0])
    np.random.shuffle(train_index)
    x_train = x_train[train_index, :, :, :]
    y_train = y_train[train_index]

    #y_train = np_utils.to_categorical(y_train, 7)
    #y_val = np_utils.to_categorical(y_val, 7)
    #y_all = np_utils.to_categorical(y_all, 7)

    return x_train, y_train, x_val, y_val
