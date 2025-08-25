import os
import xml.etree.ElementTree as ET
import tensorflow as tf




def parse_face_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        conf = xmin = ymin = xmax = ymax =0
        if cls == "man":
            bndbox = obj.find('bndbox')
            conf = 1
            xmin = int(bndbox.find('xmin').text) / width
            ymin = int(bndbox.find('ymin').text) / height
            xmax = int(bndbox.find('xmax').text) / width
            ymax = int(bndbox.find('ymax').text) / height
            boxes.append([conf, xmin, ymin, xmax, ymax])
        elif cls == "noman":
            conf = xmin = ymin = xmax = ymax = 0
        
    return img_filename, boxes

def load_face_dataset(voc_root, img_size=(60, 80), split_file=None):
    annotations_dir = os.path.join(voc_root, "Annotations")
    images_dir =  os.path.join(voc_root, "JPEGImages")

    if split_file:
        with open(split_file, "r") as f:
            image_ids = [x.strip() for x in f.readlines()]
        annotation_files = [os.path.join(annotations_dir, f"{img_id}.xml") for img_id in image_ids]
    else:
        annotation_files = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    dataset_ = []
    for xml_path in annotation_files:
        img_name, boxes = parse_face_xml(xml_path)
        if len(boxes) == 0:
            continue
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            dataset_.append((img_path, boxes))
    print(f"Loaded {len(dataset_)} face samples.")

    def _load_samples(img_path, boxes):
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_image(img_raw, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        img = (img / 255.0 - 0.5) * 2