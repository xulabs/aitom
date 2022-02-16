import os

import mrcfile
import numpy as np

from data_config import class_10, class_50, class_100, class_dict


def mapping_types(num_classes):
    if num_classes == 10:
        types = class_10
    elif num_classes == 50:
        types = class_50
    elif num_classes == 100:
        types = class_100
    else:
        raise ValueError("num_classes only support (10, 50, 100)")

    mapping_pairs = {type: idx for idx, type in enumerate(types)}
    return mapping_pairs


def preprocess_data(
    data_dir="data/",
    num_classes=50,
    target_snr_type="SNR005",
    num_records=500,
    train_percent=0.96,
    validate_percent=0.0,
    normalization=True
):

    hashing_tpyes = mapping_types(num_classes)
    storage_mat = np.zeros([len(hashing_tpyes), num_records, 32, 32, 32])
    data_dir = os.path.join(data_dir, class_dict[num_classes])

    for type_label, idx_type_label in hashing_tpyes.items():
        mrc_dir = os.path.join(data_dir, target_snr_type, type_label, "subtomogram_mrc")
        for num_record in range(num_records):
            try:
                with mrcfile.open(
                    os.path.join(mrc_dir, f"tomotarget{num_record}.mrc")
                ) as mrc:
                    storage_mat[idx_type_label, num_record, ...] = mrc.data
            except Exception as e:
                print(e.args, f"{type_label}/tomotarget{num_record}.mrc")
        print(f"finish processing {mrc_dir} \t index: {idx_type_label}")

    if normalization:
        storage_mat = standard_normalizaiton(storage_mat, 1)
        print(f"finish normalizing")

    output_dict = {"train": [], "valid": [], "test": []}

    train_end = int(train_percent * num_records)
    validate_end = int(validate_percent * num_records) + train_end
    train = storage_mat[:, :train_end, ...]
    #train = storage_mat[:, :200, ...]
    valid = storage_mat[:, train_end:validate_end, ...]
    test = storage_mat[:, validate_end:, ...]

    for key_type, output_mat in output_dict.items():
        for idx_type in range(len(hashing_tpyes)):
            cur_type = eval(key_type)[idx_type].reshape(-1, 1, 32, 32, 32)
            for row_idx in range(cur_type.shape[0]):
                output_mat.append([cur_type[row_idx], idx_type])

    return output_dict


def generate_seq(data_seq, sample_rate=1):
    data_seq = data_seq[: int(np.floor(len(data_seq) * sample_rate))]
    return data_seq


def train_validate_test_split(data, train_percent=0.6, validate_percent=0.2):
    total_seq = len(data)
    data = np.array(data)
    perm = np.random.permutation(total_seq)
    train_end = int(train_percent * total_seq)
    validate_end = int(validate_percent * total_seq) + train_end
    train = data[perm[:train_end]]
    valid = data[perm[train_end:validate_end]]
    test = data[perm[validate_end:]]
    return train, valid, test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def standard_normalizaiton(data, dim):
    normalized = (data - 0.03) / 13.71    # 50 classes of SNR005
    return normalized
