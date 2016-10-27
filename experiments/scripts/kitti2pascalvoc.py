import sys
import argparse
from xml.dom.minidom import Document
import cv2, os
import glob
import xml.etree.ElementTree as ET
import shutil

def generate_xml(name, lines, img_size = (370, 1224, 3), class_sets = ('pedestrian', 'car', 'cyclist')):
    """
    Write annotations into voc xml format.
    Examples:
        In: 0000001.txt
            cls        truncated    occlusion   angle   boxes                         3d annotation...
            Pedestrian 0.00         0           -0.20   712.40 143.00 810.73 307.92   1.89 0.48 1.20 1.84 1.47 8.41 0.01
        Out: 0000001.xml
            <annotation>
                <folder>VOC2007</folder>
	            <filename>000001.jpg</filename>
	            <source>
	            ...
	            <object>
                    <name>Pedestrian</name>
                    <pose>Left</pose>
                    <truncated>1</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>x1</xmin>
                        <ymin>y1</ymin>
                        <xmax>x2</xmax>
                        <ymax>y2</ymax>
                    </bndbox>
            	</object>
            </annotation>
    :param name: stem name of an image, example: 0000001
    :param lines: lines in kitti annotation txt
    :param img_size: [height, width, channle]
    :param class_sets: ('Pedestrian', 'Car', 'Cyclist')
    :return:
    """

    doc = Document()

    def append_xml_node_attr(child, parent = None, text = None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    img_name=name+'.jpg'

    # create header
    annotation = append_xml_node_attr('annotation')
    append_xml_node_attr('folder', parent = annotation, text='KITTI')
    append_xml_node_attr('filename', parent = annotation, text=img_name)
    source = append_xml_node_attr('source', parent=annotation)
    append_xml_node_attr('database', parent=source, text='KITTI')
    append_xml_node_attr('annotation', parent=source, text='KITTI')
    append_xml_node_attr('image', parent=source, text='KITTI')
    append_xml_node_attr('flickrid', parent=source, text='000000')
    owner = append_xml_node_attr('owner', parent=annotation)
    append_xml_node_attr('url', parent=owner, text = 'http://www.cvlibs.net/datasets/kitti/index.php')
    size = append_xml_node_attr('size', annotation)
    append_xml_node_attr('width', size, str(img_size[1]))
    append_xml_node_attr('height', size, str(img_size[0]))
    append_xml_node_attr('depth', size, str(img_size[2]))
    append_xml_node_attr('segmented', parent=annotation, text='0')

    # create objects
    for line in lines:
        splitted_line = line.strip().lower().split()
        hard = 1 if is_hard(splitted_line) else 0
        truncted = 0 if float(splitted_line[1]) < 0.1 else 1
        if splitted_line[0] in class_sets:
            obj = append_xml_node_attr('object', parent=annotation)
            append_xml_node_attr('name', parent=obj, text=splitted_line[0].lower())
            append_xml_node_attr('pose', parent=obj, text='Left')
            append_xml_node_attr('truncated', parent=obj, text=str(truncted))
            append_xml_node_attr('difficult', parent=obj, text=str(int(hard)))
            bb = append_xml_node_attr('bndbox', parent=obj)
            append_xml_node_attr('xmin', parent=bb, text=str(int(float(splitted_line[4]))))
            append_xml_node_attr('ymin', parent=bb, text=str(int(float(splitted_line[5]))))
            append_xml_node_attr('xmax', parent=bb, text=str(int(float(splitted_line[6]))))
            append_xml_node_attr('ymax', parent=bb, text=str(int(float(splitted_line[7]))))

    return  doc

def is_hard(splitted_line):
    cls = str(splitted_line[0])
    truncation = float(splitted_line[1])
    occlusion = float(splitted_line[2])
    angle = float(splitted_line[3])
    x1 = float(splitted_line[4])
    y1 = float(splitted_line[5])
    x2 = float(splitted_line[6])
    y2 = float(splitted_line[7])

    # the rest annotations are for 3d.

    # Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
    # Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
    # Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
    hard = False
    if y2 - y1 < 25 and occlusion >= 2:
        hard = True
        return hard
    if truncation > 0.5:
        hard = True
        return hard

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert KITTI dataset into Pascal voc format')
    parser.add_argument('--kitti', dest='kitti',
                        help='path to kitti root',
                        default='./data/KITTI', type=str)
    parser.add_argument('--out', dest='outdir',
                        help='path to voc-kitti',
                        default='./data/KITTIVOC', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)

    args = parser.parse_args()
    return args

def build_voc_dirs(outdir):
    """
    Build voc dir structure:
        VOC2007
            |-- Annotations
                    |-- ***.xml
            |-- ImageSets
                    |-- Layout
                            |-- [test|train|trainval|val].txt
                    |-- Main
                            |-- class_[test|train|trainval|val].txt
                    |-- Segmentation
                            |-- [test|train|trainval|val].txt
            |-- JPEGImages
                    |-- ***.jpg
            |-- SegmentationClass
                    [empty]
            |-- SegmentationObject
                    [empty]
    """
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'ImageSets'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
    mkdir(os.path.join(outdir, 'JPEGImages'))
    mkdir(os.path.join(outdir, 'SegmentationClass'))
    mkdir(os.path.join(outdir, 'SegmentationObject'))

    return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'ImageSets', 'Main')

if __name__ == '__main__':
    args = parse_args()

    _kittidir = args.kitti
    _outdir = args.outdir
    _dest_label_dir, _dest_img_dir, _dest_set_dir = build_voc_dirs(_outdir)

    # for kitti only provides training labels
    for dset in ['train']:

        _labeldir = os.path.join(_kittidir, 'training', 'label_2')
        _imagedir = os.path.join(_kittidir, 'training', 'image_2')

        class_sets = ('pedestrian', 'car', 'cyclist')
        fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets ]
        class_sets_dict = dict((k, i) for i, k in enumerate(class_sets))
        ftrain = open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')

        files = glob.glob(os.path.join(_labeldir, '*.txt'))
        files.sort()
        for file in files:
            path, basename = os.path.split(file)
            stem, ext = os.path.splitext(basename)
            with open(file, 'r') as f:
                lines = f.readlines()
            img_file = os.path.join(_imagedir, stem + '.png')
            img = cv2.imread(img_file)
            img_size = img.shape
            cv2.imwrite(os.path.join(_dest_img_dir, stem + '.jpg'), img)
            doc = generate_xml(stem, lines, img_size, class_sets)

            xmlfile = os.path.join(_dest_label_dir, stem + '.xml')
            with open(xmlfile, 'w') as f:
                f.write(doc.toprettyxml(indent='	'))

            ftrain.writelines(stem + '\n')

            # build [cls_train.txt]
            # Car_train.txt: 0000xxx [1 | -1]
            tree = ET.parse(xmlfile)
            objs = tree.findall('object')
            # non_diff_objs = [ obj for obj in objs if int(obj.find('difficult').text) == 0]
            # objs = non_diff_objs
            num_objs = len(objs)
            objnames = []
            for i, obj in enumerate(objs):
                class_name = obj.find('name').text.strip()
                objnames.append(class_name)
            objnames = set(objnames)
            for cls in objnames:
                fs[class_sets_dict[cls]].writelines(stem + ' 1\n')
            for cls in class_sets:
                if cls not in objnames:
                    fs[class_sets_dict[cls]].writelines(stem + ' -1\n')

            if int(stem) % 100 == 0:
                print(file)

        for f in fs:
            f.close()
        ftrain.close()
        shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'val.txt'))
        shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'trainval.txt'))
        for cls in class_sets:
            shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
                            os.path.join(_dest_set_dir, cls + '_trainval.txt'))
            shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
                            os.path.join(_dest_set_dir, cls + '_val.txt'))
