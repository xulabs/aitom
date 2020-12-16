from __future__ import division
from .ssnr3d import SSNR3D
import pickle
import time
import random
from pyExcelerator import *
from xlrd import open_workbook
from xlutils.copy import copy

# beta used in F-meature
BETA = 0.5
# ratio = number of homogeneous image / number of heterogeneous image
RATIO = 0.1


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


def matching(ratio):
    imgs = {}
    masks = {}
    result = []
    true_image_set = []

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

    # img_siz = (40, 40)
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
    center = center_num1
    result = [center]
    temp = [center]

    while True:
        distance = []
        for i in range(len(img_set)):
            if i != center:
                temp.append(i)
                s.set_img_set(temp)
                distance.append(s.get_fsc_sum())
            else:
                distance.append(-1)
        center = distance.index(max(distance))
        temp = [center]

        if center in result:
            break
        else:
            result.append(center)

    center = center_num2
    if center not in result:
        while True:
            distance = []
            for i in range(len(img_set)):
                if i != center:
                    temp.append(i)
                    s.set_img_set(temp)
                    distance.append(s.get_fsc_sum())
                else:
                    distance.append(-1)
            center = distance.index(max(distance))
            temp = [center]

            if center in result:
                break
            else:
                result.append(center)

    end = time.clock()

    resolution = s.get_fsc_sum()
    t = end - start
    measure = cal_accuracy(result, true_image_set)
    print("Resolution = " + str(resolution))
    print("Result: " + ", ".join(result))

    # sheet = 2
    sheet = 0
    start_y = 1
    # start_y = 1 + int(10*ratio-1)

    write_to_excel(sheet, 0, start_y, ratio)
    write_to_excel(sheet, 1, start_y, resolution)
    write_to_excel(sheet, 2, start_y, measure[0])
    write_to_excel(sheet, 3, start_y, measure[1])
    write_to_excel(sheet, 4, start_y, measure[2])
    write_to_excel(sheet, 5, start_y, t)


# x->col, y->row
def write_to_excel(sheet, x, y, value):
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
    matching(RATIO)


if __name__ == '__main__':
    main()
