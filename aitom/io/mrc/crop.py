import mrcfile
from .. import file as io_file


def crop_mrc(mrc_path, crop_path, x=0, y=0, z=0, dx=100, dy=100, dz=100, print_header_diff=False):
    """
    Crop specified part of a 3d mrc file, position cropped: mrc.data[x:x+dx, y:y+dy, z:z+dz].
    
    Note that the axis order of mrc data is (x, y, z), different from that downstream tasks use, eg. read_mrc_data
    and imod: (z, y, x).

    Arguments:
        mrc_path: source mrc file path
        crop_path: destination path of cropped mrc file

    Keyword Arguments:
        x {int}: lowerbound x coordinate to crop (default: {0})
        y {int}: lowerbound y coordinate to crop (default: {0})
        z {int}: lowerbound z coordinate to crop (default: {0})
        dx {int}: length of x to crop (default: {100})
        dy {int}: length of y to crop (default: {100})
        dz {int}: length of z to crop (default: {100})
        print_header_diff {bool}: whether to print difference between the cropped and original (default: {False})
    """
    # Use mmap for faster reading large mrcfile
    with mrcfile.mmap(mrc_path, mode='r') as mrc, mrcfile.new(crop_path) as mrc_crop:
        # set_data automatically syncs header info with data
        mrc_crop.set_data(mrc.data[x:x + dx, y:y + dy, z:z + dz])
        mrc_crop.voxel_size = mrc.voxel_size
    mrcfile.validate(crop_path)

    # Print header diff
    if print_header_diff:
        mrc_header = io_file.read_mrc_header(mrc_path)
        crop_header = io_file.read_mrc_header(crop_path)
        diffs = []
        for (k1, v1), (k2, v2) in zip(mrc_header.items(), crop_header.items()):
            if k1 == k2 and v1 != v2:
                if isinstance(v1, dict):
                    assert len(v1) == len(v2), "Different dict size: {}, {}".format(len(v1), len(v2))
                    diff_dict = {k: (v1[k], v2[k]) for k in v1 if k in v2 and v1[k] != v2[k]}
                    print("diff key: ", k1, "\ndiff_dict:\n", diff_dict)
                else:
                    print("diff key: ", k1, "\n", v1, "\n", v2, "\n")
                diffs.append((v1, v2))
        print("# diffs: ", len(diffs))


if __name__ == "__main__":
    # Try your own example here
    mrc_path = 'IS002_291013_005.mrc'
    crop_path = 'IS002_291013_005_crop_100.mrc'
    crop_mrc(mrc_path, crop_path)
