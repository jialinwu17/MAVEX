import numpy as np

import copy


def read_features(path, num_boxes = 100):
    pickled_features = np.load(path, allow_pickle=True).item()
    image_h = int(pickled_features["image_height"])
    image_w = int(pickled_features["image_width"])
    features = pickled_features["features"].reshape(-1, 2048)[:num_boxes]
    boxes = pickled_features["bbox"].reshape(-1, 4)[:num_boxes]
    num_boxes = features.shape[0]
    g_feat = np.sum(features, axis=0) / num_boxes
    num_boxes = num_boxes + 1
    features = np.concatenate(
        [np.expand_dims(g_feat, axis=0), features], axis=0
    )

    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = boxes
    image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0])
            / (float(image_w) * float(image_h))
    )

    image_location_ori = copy.deepcopy(image_location)
    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)

    g_location = np.array([0, 0, 1, 1, 1])
    image_location = np.concatenate(
        [np.expand_dims(g_location, axis=0), image_location], axis=0
    )

    g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
    image_location_ori = np.concatenate(
        [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
    )

    return features, num_boxes, image_location, image_location_ori
