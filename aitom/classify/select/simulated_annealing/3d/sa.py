from __future__ import division
from ssnr3d import SSNR3D
import pickle
import time
import random
import math
from pyExcelerator import *
from xlrd import open_workbook
from xlutils.copy import copy

BETA = 0.5 # beta used in F-meature
RATIO = 0.1 # ratio = number of homogeneous image / number of heterogeneous image
ITERATION = 20 # number of iteration


def img(fileName):
    pkl_file = open(fileName, 'rb')
    images = {}
    images = pickle.load(pkl_file)
    pkl_file.close()
    img = []
    mask = []
    data = []
    for i in images.values():
        img.append(i['v'])
        mask.append(i['m'])
    data.append(img)
    data.append(mask)
    return data


# don't know the number of target images
def simulated_annealing(ratio):
    imgs = {}
    masks = {}

    t = [] # time
    measure = []
    true_image_set = []
    resolution = [] #record the resolution after every iteration

    img_set = []
    mask = []
    img_true = img("true__out__images.pickle")[0]
    mask_true = img("true__out__images.pickle")[1]
    img_rotate = img("rotation_var__out__images.pickle")[0]
    mask_rotate = img("rotation_var__out__images.pickle")[1]

    for i in range(100):
        img_set.append(img_rotate[i])
        mask.append(mask_rotate[i])
        if i < int(100 * ratio):
            img_set.append(img_true[i])
            mask.append(mask_true[i])
            true_image_set.append(len(img_set) - 1)

    #img_siz = (40, 40)
    for i in range(len(img_set)):
        imgs[i] = img_set[i]
        masks[i] = mask[i]

    s = SSNR3D(imgs, masks)

    start = time.clock()

    while True:
        result = []
        center_num1 = random.randrange(0, len(img_set), 1)
        center_num2 = random.randrange(0, len(img_set), 1)
        if center_num1 != center_num2:
            result.append(center_num1)
            result.append(center_num2)
            s.set_img_set(result)
            e = s.get_fsc_sum()
            if e > 6.8:
                break
    print(result)

    s0 = center_num2
    sn = s0
    k = 0
    kmax = 100 + int(100 * ratio)

    # first iteration

    while (k < kmax) and (len(result) < kmax // 2):
        while True:
            sn = sn + 1
            if sn == kmax:
                sn = 0
            if sn not in result:
                break

        s.add_to_set(sn)
        en = s.get_fsc_sum()
        s.remove_from_set(sn)

        if en > e:
            result.append(sn)
            s.set_img_set(result)
            e = en
        elif math.exp(-abs(en - e) /
                      (k + 1)) < random.random(): # after testing, choose 350 (need >50)
            result.append(sn)
            s.set_img_set(result)
            #e = en
        k += 1
    s_record = sn #record "sn" and when next iteration, start from sn

    for sn in result:
        s.remove_from_set(sn)
        en = s.get_fsc_sum()
        s.add_to_set(sn)
        if en > e:
            result.remove(sn)
            s.set_img_set(result)
            e = en

    resolution.append(e)
    measure.append(cal_accuracy(result, true_image_set))

    end = time.clock()
    t.append(end - start)
    print("Resolution = " + str(e))

    # iterations

    for i in range(ITERATION - 1):
        k = 0
        sn = s_record
        while (k < kmax) and (len(result) < kmax // 2):
            while True:
                sn += 1
                if sn == kmax:
                    sn = 0
                if sn not in result:
                    break

            s.add_to_set(sn)
            en = s.get_fsc_sum()
            s.remove_from_set(sn)
            if en > e:
                result.append(sn)
                s.set_img_set(result)
                e = en
            elif math.exp(-abs(en - e) / (k + 1)) < random.random():
                result.append(sn)
                s.set_img_set(result)
                #e = en
            k += 1
        s_record = sn

        for sn in result:
            s.remove_from_set(sn)
            en = s.get_fsc_sum()
            s.add_to_set(sn)
            if en > e:
                result.remove(sn)
                s.set_img_set(result)
                e = en
        resolution.append(e)
        measure.append(cal_accuracy(result, true_image_set))

        end = time.clock()
        t.append(end - start)

    print("Resolution = " + str(e))
    print("Time = " + str(end - start))
    print("Result: " + ", ".join(result))

    #start_y = 1 + int(10*ratio-1)*(ITERATION+1)
    start_y = 1
    sheet = 0

    for i in range(start_y, ITERATION + start_y):
        write_to_excel(sheet, 1, i, resolution[i - start_y])
        write_to_excel(sheet, 2, i, measure[i - start_y][0])
        write_to_excel(sheet, 3, i, measure[i - start_y][1])
        write_to_excel(sheet, 4, i, measure[i - start_y][2])
        write_to_excel(sheet, 5, i, t[i - start_y])
        if i == start_y:
            write_to_excel(sheet, 0, i, ratio)


def write_to_excel(sheet, x, y, value): #x->col, y->row
    rb = open_workbook('record.xls')
    rs = rb.sheet_by_index(sheet)
    wb = copy(rb)
    ws = wb.get_sheet(sheet)
    ws.write(y, x, value)
    wb.save('record.xls')


def cal_accuracy(result, true_image_set):
    return_list = []
    n_right = 0
    n_wrong = 0
    for i in result:
        if i in true_image_set:
            n_right += 1
        else:
            n_wrong += 1
    precision = n_right / len(result)
    recall = n_right / len(true_image_set)
    print("precision = " + str(precision * 100) + " %")
    print("recall = " + str(recall * 100) + " %")
    if (precision != 0.0) and (recall != 0.0):
        f = (1 + pow(BETA, 2)) * (precision * recall) / (precision * pow(BETA, 2) + recall)
    else:
        f = 0.0
    return_list.append(precision)
    return_list.append(recall)
    return_list.append(f)

    return return_list


def main():
    simulated_annealing(RATIO)


if __name__ == '__main__':
    main()
