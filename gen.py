import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 加载数据
def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data.get('images', [])}
    categories = data.get('categories', [])
    annotations = data.get('annotations', [])

    return annotations, categories, images


# 归一化关键点坐标
def normalize_keypoints(keypoints, image_width, image_height):
    keypoints = np.array(keypoints).reshape(-1, 3)[:, :2].astype(np.float64)
    keypoints[:, 0] /= image_width
    keypoints[:, 1] /= image_height
    return keypoints


# 计算骨架连接的距离和角度
def calculate_distance_and_angle(keypoints, skeleton):
    distances = []
    angles = []

    # 计算每个连接的距离和角度
    for link in skeleton:
        pt1_idx = link[0] - 1
        pt2_idx = link[1] - 1

        pt1 = np.array([keypoints[pt1_idx * 2], keypoints[pt1_idx * 2 + 1]])  # (x, y) 坐标
        pt2 = np.array([keypoints[pt2_idx * 2], keypoints[pt2_idx * 2 + 1]])  # (x, y) 坐标

        # 检查是否有一个关键点为 (0, 0)，如果是，则距离和角度设为 0
        if np.array_equal(pt1, [0, 0]) or np.array_equal(pt2, [0, 0]):
            dist = 0
            angle_deg = 0
        else:
            # 计算距离
            dist = np.linalg.norm(pt2 - pt1)
            # 计算与水平线（x轴）之间的角度
            angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            # 将角度转换为度
            angle_deg = np.degrees(angle)

        distances.append(dist)
        angles.append(angle_deg)

    return distances, angles


# 提取骨架特征
def extract_skeleton_features(annotation, skeleton, images):
    if 'keypoints' not in annotation:
        return None

    keypoints = annotation['keypoints']
    image_id = annotation['image_id']
    image_info = images.get(image_id)

    if image_info is None:
        return None

    # 使用 bbox 的宽度和高度替代原始的图像尺寸
    bbox = annotation['bbox']  # 获取 bbox
    bbox_width, bbox_height = bbox[2], bbox[3]  # bbox 格式是 [x, y, width, height]
    image_width, image_height = bbox_width, bbox_height

    keypoints = normalize_keypoints(keypoints, image_width, image_height)
    distances, angles = calculate_distance_and_angle(keypoints.flatten(), skeleton)
    scaler1, scaler2 = MinMaxScaler(), MinMaxScaler()

    distances = np.array(distances).reshape(-1, 1)
    angles = np.array(angles).reshape(-1, 1)

    # 对数据进行缩放
    distances_scaled = scaler1.fit_transform(distances).reshape(-1)
    angles_scaled = scaler2.fit_transform(angles).reshape(-1)

    return distances_scaled + angles_scaled


# 聚类分析
def perform_clustering(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans


# 使用 t-SNE 降维
def reduce_dimensions_with_tsne(features, n_components=2, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    reduced_features = tsne.fit_transform(features)
    return reduced_features


# 获取聚类数据并保存为 JSON
import numpy as np

# 将数据转换为 Python 内建类型
def convert_to_builtin_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # 转换为 Python 基本类型
    elif isinstance(obj, dict):
        return {convert_to_builtin_type(key): convert_to_builtin_type(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(item) for item in obj]
    else:
        return obj

# 获取聚类数据并保存为 JSON
def save_cluster_data_to_json(filename, output_json_filename):
    annotations, categories, images = load_data(filename)
    skeleton = categories[0]['skeleton']

    # 过滤掉没有 'keypoints' 的条目
    annotations = [annotation for annotation in annotations if 'keypoints' in annotation]
    features = []
    for annotation in annotations:
        feature_vector = extract_skeleton_features(annotation, skeleton, images)
        if feature_vector is not None:
            features.append(feature_vector)

    features = np.array(features)

    # 使用 t-SNE 降维
    reduced_features = reduce_dimensions_with_tsne(features)

    # 聚类分析
    labels, kmeans = perform_clustering(features)

    cluster_data = {}
    for idx, label in enumerate(labels):
        image_id = annotations[idx]['image_id']
        file_name = images.get(image_id)['file_name']
        bbox = annotations[idx]['bbox']
        x, y = reduced_features[idx]

        if label not in cluster_data:
            cluster_data[label] = []

        cluster_data[label].append({
            'image_id': image_id,
            'file_name': file_name,
            'x': round(x, 3),  # 保留三位小数
            'y': round(y, 3),  # 保留三位小数
            'bbox': bbox
        })

    # 将数据转换为标准 Python 类型
    cluster_data = convert_to_builtin_type(cluster_data)

    # 保存到 JSON 文件
    with open(output_json_filename, 'w') as f:
        json.dump(cluster_data, f, indent=4)

    print(f"Cluster data saved to {output_json_filename}")



# 调用此函数将数据保存为 JSON
if __name__ == '__main__':
    input_filename = 'person_keypoints_train.json'  # 你的原始 JSON 文件路径
    output_json_filename = 'cluster_data.json'  # 保存聚类数据的 JSON 文件路径
    save_cluster_data_to_json(input_filename, output_json_filename)
