import json
import cv2
import os
import os.path as osp
from shutil import copyfile
from argparse import ArgumentParser
import sys

def sure_dir_exists(path):
    p = osp.realpath(path)
    if not osp.isdir(p):
        os.makedirs(p)
    return path

def parse_args():
    parser = ArgumentParser(description=
            'converts coco detection data for YOLO training')
    parser.add_argument('--labels', help='file with coco labels',
            required=True)
    parser.add_argument('--images', help='path to coco images',
            required=True)
    parser.add_argument('--output', help='path to write results',
            required=True)
    parser.add_argument('--num_elems', help=
            'Number of elements to output. Default is None, i.e. all',
            default=None, required=False)
    parser.add_argument('--visualize', help='Visualize data',
            action='store_true')

    return parser.parse_args(sys.argv[1:])

if __name__=='__main__':
    args = parse_args()
    labels_filename = args.labels
    img_dir = args.images
    output_dir = args.output
    num_elems = args.num_elems

    res_img_dir = sure_dir_exists(osp.join(output_dir, 'images'))
    res_label_dir = sure_dir_exists(osp.join(output_dir, 'labels'))

    img2label_fn = lambda fn: osp.join(res_label_dir,
            '{0}.txt'.format(osp.splitext(osp.basename(fn))[0]))

    with open(labels_filename, 'r') as f:
        data = json.load(f)

    # category_id to index mapping
    cat_id2ind = {el['id']: i for i, el in enumerate(data['categories'])}

    frame_id_array = [(el['id'], el['width'], el['height'], el['file_name'])
            for el in data['images']]
    if not num_elems is None:
        frame_id_array = frame_id_array[:int(num_elems)]

    for frame_id, width, height, filename in frame_id_array:
        frame_annotations = [(el['bbox'], cat_id2ind[el['category_id']]) for el in data['annotations']
                if el['image_id'] == frame_id]
        full_fn = osp.join(img_dir, filename)
        copyfile(full_fn, osp.join(res_img_dir, filename))
        with open(img2label_fn(filename), 'w') as f:
            for box, category in frame_annotations:
                f.write('{} {} {} {} {}\n'.format(
                    category,
                    (box[0] + (box[2] - 1) / 2) / float(width),
                    (box[1] + (box[3] - 1) / 2) / float(height),
                    box[2] / float(width),
                    box[3] / float(height)))

    if args.visualize:
        cat_ind2name = [el['name'] for el in data['categories']]
        filenames = [fn for fn in os.listdir(res_img_dir)]
        for fn in filenames:
            full_fn = osp.join(res_img_dir, fn)
            img = cv2.imread(full_fn)
            frame_annotations = [[float(el) for el in obj.split(' ')]
                    for obj in open(img2label_fn(fn), 'r').read().strip().split('\n')]
            for category, x, y, w, h in frame_annotations:
                tl = tuple(map(lambda c, s, f:
                    int(c * f - (s * f - 1) / 2), (x, y), (w, h), img.shape[1::-1]))
                br = tuple(map(lambda c, s, f:
                    int(c * f + (s * f - 1) / 2), (x, y), (w, h), img.shape[1::-1]))
                cv2.rectangle(img, tl, br, (0, 255, 0), 2)
                br_1 = (br[0], tl[1] + 12)
                cv2.rectangle(img, tl, br_1, (0, 255, 0), -1)
            for category, x, y, w, h in frame_annotations:
                tl = tuple(map(lambda c, s, f:
                    int(c * f - (s * f - 1) / 2), (x, y), (w, h), img.shape[1::-1]))
                tl_1 = (tl[0], tl[1] + 11)
                cv2.putText(img, cat_ind2name[int(category)], tl_1,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(filename, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
