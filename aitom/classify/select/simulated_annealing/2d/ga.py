from .ssnr2d import SSNR2D
import pickle
import time
import random
from random import choice
from pyExcelerator import *
from xlrd import open_workbook
from xlutils.copy import copy

ITERATION = 10      # number of iteration
NUM_FIRST_G = 40    # number of candidate in the first generation (after testing)
BETA = 0.5          # beta used in F-meature
RATIO = 0.1         # ratio = number of homogeneous image / number of heterogeneous image


def img(fileName):
    pkl_file = open(fileName, 'rb')
    images = {}
    images = pickle.load(pkl_file)
    pkl_file.close()
    return images


def find_img_index(vector):
    find = 1
    index = [i for i, j in enumerate(vector) if j == find]
    return index


def genetic_algorithm(ratio):
    # img_siz = (40, 40)
    dimension = 100 + int(100 * ratio)
    maxi = dimension // 2

    imgs = {}
    img_set = []
    # time
    t = []
    measure = []
    true_image_set = []
    # record the resolution after every iteration
    resolution = []

    img_true = img("true__out__images.pickle")
    img_rotate = img("rotation_var__out__images.pickle")

    for i in range(100):
        img_set.append(img_rotate[i])
        if i < int(100 * ratio):
            img_set.append(img_true[i])
            true_image_set.append(len(img_set) - 1)

    for i in range(dimension):
        imgs[i] = img_set[i]
    s = SSNR2D(imgs)

    start = time.clock()

    img_num = range(dimension)
    g_zero = [0] * dimension

    initial = []
    generation = []
    # nest generation
    generation_1 = []
    result_list = []

    # At first, 10 candidate solutions. Each solution has 200 dimensions.
    for i in range(NUM_FIRST_G):
        initial.append(random.sample(img_num, maxi))
    for i in range(NUM_FIRST_G):
        for j in range(maxi):
            g_zero[initial[i][j]] = 1
        generation.append(g_zero)
        g_zero = []
        g_zero = [0] * dimension

    iteration_list = []
    temp = []

    for i in range(NUM_FIRST_G):
        s.set_img_set(find_img_index(generation[i]))
        temp.append(s.get_fsc_sum())

    iteration_list.append(max(temp))
    result = find_img_index(generation[temp.index(max(temp))])
    result_list.append(result)

    end = time.clock()

    # first iteration
    resolution.append(max(temp))
    t.append(end - start)
    measure.append(cal_accuracy(result, true_image_set))
    print("resolution = " + str(max(temp)))

    # max number of iterations
    for i in range(ITERATION - 1):
        # There are always a half of this generation chosen to be alive and a half mutated
        temp = []
        # natural selection
        for j in range(len(generation)):
            s.set_img_set(find_img_index(generation[j]))
            temp.append(s.get_fsc_sum())

        temp_best = []
        for j in range(len(generation) // 2):
            temp_best.append(generation[temp.index(max(temp))])
            temp[temp.index(max(temp))] = 0

        # mutation process(crossover operation)
        P_new = []
        while len(P_new) < len(temp_best):
            P = random.sample(temp_best, 2)
            pl1 = P[0][:dimension // 2]
            pr1 = P[0][dimension // 2:]
            pl2 = P[1][:dimension // 2]
            pr2 = P[1][dimension // 2:]
            P_new1 = pl1 + pr2
            P_new2 = pl2 + pr1
            if random.random() < 0.5:
                ran = random.randint(0, dimension - 1)
                if P_new1[ran] == 0:
                    P_new1[ran] = 1
                else:
                    P_new1[ran] = 0
                ran = random.randint(0, dimension - 1)
                if P_new2[ran] == 0:
                    P_new2[ran] = 1
                else:
                    P_new2[ran] = 0

            P_new.append(P_new1)
            P_new.append(P_new2)

        generation_1 = P_new + generation

        # remove redundant images
        temp = []
        for j in generation_1:
            if len(find_img_index(j)) > maxi:
                temp.append(j)
        for j in temp:
            generation_1.remove(j)

        best = 0
        for j in generation_1:
            s.set_img_set(find_img_index(j))
            best_new = s.get_fsc_sum()
            if best_new > best:
                best = best_new
                result = find_img_index(j)
        # print(best)
        result_list.append(result)
        iteration_list.append(best)

        end = time.clock()

        resolution.append(best)
        t.append(end - start)
        measure.append(cal_accuracy(result, true_image_set))
        print("resolution = " + str(best))

        generation = generation_1
        generation_1 = []

    print("Result: " + ", ".join(result))

    # start_y = 1 + int(10 * ratio - 1) * 11
    # sheet = 1
    sheet = 0
    start_y = 1

    for i in range(start_y, ITERATION + start_y):
        write_to_excel(sheet, 1, i, resolution[i - start_y])
        write_to_excel(sheet, 2, i, measure[i - start_y][0])
        write_to_excel(sheet, 3, i, measure[i - start_y][1])
        write_to_excel(sheet, 4, i, measure[i - start_y][2])
        write_to_excel(sheet, 5, i, t[i - start_y])
        if i == start_y:
            write_to_excel(sheet, 0, i, ratio)


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
    genetic_algorithm(RATIO)


if __name__ == '__main__':
    main()
