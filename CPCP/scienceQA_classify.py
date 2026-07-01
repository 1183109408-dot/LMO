import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
from data_deal.scienceQA_load import load_and_process_json

# 设置随机种子
np.random.seed(42)

# 参数设置
TRAIN_JSON_PATH = r"E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\datas\scienceQA\train4.json"
TEST_JSON_PATH = r"E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\datas\scienceQA\test4.json"
MODEL_PATH = r"E:\project\OSAI\model\bert-base-cased"
OUTPUT_PATH = r"E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\classifier"
GPU = 0
FEATURE_DIM = 768
VARIANCE_THRESHOLD = 0.96
N_COMPONENTS = 80  # LDA 降维维度
LAMBDA_B = 0.4
LAMBDA_W = 0.4

class BertEmbedding(nn.Module):
    def __init__(self, model_path):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
    
    def forward(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.cuda(GPU) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]

def apply_pca_regularized_lda(features, labels, lambda_b, lambda_w, n_components):
    """
    应用 PCA 和正则化 LDA 降维，并计算 LDA 降维后的类别均值向量
    """
    features_np = features.detach().cpu().numpy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算原始特征空间的类别均值（768 维）
    unique_labels = sorted(list(set(labels)))
    class_centroids_raw = {}
    for s in unique_labels:
        small_features = features_np[np.array(labels) == s]
        mu_s = np.mean(small_features, axis=0)
        class_centroids_raw[s] = mu_s

    # PCA 降维
    pca = PCA(n_components=VARIANCE_THRESHOLD)
    pca_features = pca.fit_transform(features_scaled)
    n_components_pca = pca_features.shape[1]
    print(f"PCA 降维维度: {n_components_pca}, 保留方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 计算类间散度矩阵 S_b
    n_samples = len(features_np)
    overall_mean = np.mean(pca_features, axis=0)
    S_b = np.zeros((n_components_pca, n_components_pca))
    for s in unique_labels:
        small_features = pca_features[np.array(labels) == s]
        N_s = len(small_features)
        mu_s = np.mean(small_features, axis=0)
        diff_s = (mu_s - overall_mean).reshape(-1, 1)
        S_b += N_s * (diff_s @ diff_s.T)
    S_b = (S_b + lambda_b * np.cov(pca_features, rowvar=False) * n_samples / (n_samples - 1)) / n_samples

    # 计算类内散度矩阵 S_w
    S_w = np.zeros((n_components_pca, n_components_pca))
    for s in unique_labels:
        small_features = pca_features[np.array(labels) == s]
        mu_s = np.mean(small_features, axis=0)
        for x in small_features:
            diff = (x - mu_s).reshape(-1, 1)
            S_w += diff @ diff.T
    S_w = (S_w + lambda_w * np.cov(pca_features, rowvar=False) * n_samples / (n_samples - 1)) / n_samples

    # 特征分解
    S_w_inv = np.linalg.inv(S_w + 1e-6 * np.eye(n_components_pca))
    eigvals, eigvecs = np.linalg.eigh(S_w_inv @ S_b)
    idx = np.argsort(eigvals)[::-1]
    n_components_lda = min(n_components, np.linalg.matrix_rank(S_b, tol=1e-4))
    W_lda = eigvecs[:, idx[:n_components_lda]]
    print(f"LDA 降维维度: {n_components_lda}")

    # 融合 PCA 和 LDA 的投影矩阵
    W_fusion = pca.components_.T @ W_lda
    reduced_features = features_scaled @ W_fusion
    reduced_features = np.nan_to_num(reduced_features, nan=0.0, posinf=0.0, neginf=0.0)

    # LDA 降维后的类别均值向量（从原始均值向量投影）
    class_label_vectors = {
        s: (scaler.transform(class_centroids_raw[s].reshape(1, -1)) @ W_fusion)[0]
        for s in class_centroids_raw
    }

    return reduced_features, W_fusion, scaler, pca, n_components_lda, class_label_vectors

def apply_transform(features, projection_matrix, scaler):
    """
    对特征应用降维变换
    """
    features_np = features.detach().cpu().numpy()
    features_scaled = scaler.transform(features_np)
    reduced_features = features_scaled @ projection_matrix
    reduced_features = np.nan_to_num(reduced_features, nan=0.0, posinf=0.0, neginf=0.0)
    return reduced_features

def train_and_evaluate_classifier(train_big_classes, test_big_classes, feature_encoder):
    """
    训练和评估分类器，并返回类别标签向量
    """
    # 提取训练集特征
    train_features = []
    train_labels = []
    for big_class in train_big_classes:
        for small_class in big_class.data:
            label = small_class.label
            for sample in small_class.data:
                features = feature_encoder([sample.quest]).cuda(GPU)
                train_features.append(features)
                train_labels.append(label)

    train_features_tensor = torch.cat(train_features, dim=0)
    train_reduced_features, projection_matrix, scaler, pca, n_components_lda, class_label_vectors = apply_pca_regularized_lda(
        train_features_tensor, train_labels, LAMBDA_B, LAMBDA_W, N_COMPONENTS
    )

    # 训练 SVM
    svm = SVC(kernel='rbf', C=10, probability=True, random_state=42)
    svm.fit(train_reduced_features, train_labels)
    train_predictions = svm.predict(train_reduced_features)
    train_accuracy = np.mean([p == t for p, t in zip(train_predictions, train_labels)]) * 100
    print(f"训练集精度: {train_accuracy:.2f}%")

    # 提取测试集特征并评估
    test_features = []
    test_labels = []
    for big_class in test_big_classes:
        for small_class in big_class.data:
            label = small_class.label
            for sample in small_class.data:
                features = feature_encoder([sample.quest]).cuda(GPU)
                test_features.append(features)
                test_labels.append(label)

    test_features_tensor = torch.cat(test_features, dim=0)
    test_reduced_features = apply_transform(test_features_tensor, projection_matrix, scaler)
    test_predictions = svm.predict(test_reduced_features)
    test_accuracy = np.mean([p == t for p, t in zip(test_predictions, test_labels)]) * 100
    print(f"测试集精度: {test_accuracy:.2f}%")

    print(f"分类器训练完成，小类数量: {len(set(train_labels))}")
    return (
        svm,
        projection_matrix,
        scaler,
        pca,
        n_components_lda,
        train_accuracy,
        test_accuracy,
        class_label_vectors
    )

def main():
    """
    主函数：加载数据、训练分类器、保存参数
    """
    print("加载训练数据和测试数据")
    train_big_classes = load_and_process_json(TRAIN_JSON_PATH)
    test_big_classes = load_and_process_json(TEST_JSON_PATH)
    if not train_big_classes or not test_big_classes:
        print("数据加载失败，程序退出")
        return

    print("初始化 BERT 嵌入模型")
    feature_encoder = BertEmbedding(MODEL_PATH).cuda(GPU)

    print("训练并评估分类器")
    (
        svm,
        projection_matrix,
        scaler,
        pca,
        n_components_lda,
        train_accuracy,
        test_accuracy,
        class_label_vectors
    ) = train_and_evaluate_classifier(
        train_big_classes, test_big_classes, feature_encoder
    )

    # 保存分类器及相关对象
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    joblib.dump(svm, os.path.join(OUTPUT_PATH, "svm_classifier.pkl"))
    np.save(os.path.join(OUTPUT_PATH, "projection_matrix.npy"), projection_matrix)
    joblib.dump(scaler, os.path.join(OUTPUT_PATH, "scaler.pkl"))
    joblib.dump(pca, os.path.join(OUTPUT_PATH, "pca.pkl"))
    np.save(os.path.join(OUTPUT_PATH, "n_components_lda.npy"), np.array(n_components_lda))
    np.save(os.path.join(OUTPUT_PATH, "lambda_b.npy"), np.array(LAMBDA_B))
    np.save(os.path.join(OUTPUT_PATH, "lambda_w.npy"), np.array(LAMBDA_W))
    joblib.dump(class_label_vectors, os.path.join(OUTPUT_PATH, "class_label_vectors.pkl"))
    print(f"分类器及降维参数已保存至: {OUTPUT_PATH}")
    print(f"最终结果 - 训练集精度: {train_accuracy:.2f}%, 测试集精度: {test_accuracy:.2f}%")
    print(f"类别标签向量数量: {len(class_label_vectors)}, 向量维度: {len(list(class_label_vectors.values())[0])}")

if __name__ == "__main__":
    main()