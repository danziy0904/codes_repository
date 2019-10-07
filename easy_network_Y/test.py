import argparse
import os

def test(args):
    workspace = args.workspace
    # print(workspace)
    # r = workspace
    print(os.getcwd())

    hdf5_path = os.path.join(workspace, 'logmel', 'development.h5')
    print(hdf5_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    #parser.add_argument('--subdir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    args = parser.parse_args()
    test(args)