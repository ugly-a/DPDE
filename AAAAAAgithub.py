import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from imblearn.over_sampling import SMOTE,KMeansSMOTE, SVMSMOTE, BorderlineSMOTE,ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score,\
    accuracy_score, recall_score,precision_score,mean_squared_error,silhouette_score,v_measure_score,pairwise_distances
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split,KFold,StratifiedShuffleSplit,cross_val_score
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn import decomposition
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks,ClusterCentroids,\
    EditedNearestNeighbours,InstanceHardnessThreshold
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LassoCV
import PyIFS
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.svm import SVC
import warnings
import numpy as np
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

class DPDE:
    def __init__(self, sub_num=100,kernel_pca_params=None, random_state=0,resampler_alpha=0.6,stay_ratio=0.8,beta=0.5):
        self.sub_num = sub_num
        self.kernel_pca_params = kernel_pca_params or {'n_components': 10, 'kernel': 'poly','fit_inverse_transform':True,'remove_zero_eig':True}
        self.random_state = random_state
        self.clf = LogisticRegression(penalty='l2',C=0.5, solver='liblinear', max_iter=10000)
        # 新增初始化参数
        self.selected_counts = None
        self.models =None
        self.majority_indices_ = None
        self.resampler_alpha=resampler_alpha
        self.model_alphas = []
        self.stay_ratio = stay_ratio
        self.beta=beta

    def resampler(self, X_train, y_train):
        counts = np.bincount(y_train)
        majority_class = np.argmax(counts)
        minority_class = 1 - majority_class
        majority_samples = X_train[y_train == majority_class]
        minority_samples = X_train[y_train == minority_class]

        # 获取原始分布统计量
        original_mean = np.mean(majority_samples, axis=0)
        original_std = np.std(majority_samples, axis=0) + 1e-8
        # 获取当前批次样本权重（与majority_indices_对齐）
        majority_mask = (y_train == majority_class)

        # 指标1：特征保持度得分（与原始分布的KL散度近似）
        z_scores = (majority_samples - original_mean) / original_std
        feature_preserve = np.exp(-0.5 * np.mean(z_scores ** 2, axis=1))

        # 指标2：动态调整近邻数（避免超过少数类样本数量）
        n_neighbors = 10
        if len(minority_samples) < n_neighbors:
            n_neighbors = len(minority_samples)
            if n_neighbors == 0:
                raise ValueError("Minority class samples are required for boundary calculation.")

        # 计算每个多数样本到最近n个少数样本的距离
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(minority_samples)
        distances, _ = nn.kneighbors(majority_samples)

        # 计算平均距离并归一化得到敏感度
        avg_distances = distances.mean(axis=1)
        min_dist, max_dist = avg_distances.min(), avg_distances.max()
        boundary_sensitivity = 1 - (avg_distances - min_dist) / (max_dist - min_dist + 1e-8)

        boundary_sensitivity_min = np.min(boundary_sensitivity)
        boundary_sensitivity_max = np.max(boundary_sensitivity)
        if boundary_sensitivity_max > boundary_sensitivity_min:
            boundary_sensitivity_normalized = (boundary_sensitivity - boundary_sensitivity_min) / (
                    boundary_sensitivity_max - boundary_sensitivity_min)
        else:
            boundary_sensitivity_normalized = np.ones_like(boundary_sensitivity)  # 如果所有值相同，则归一化为 1
        feature_preserve_min = np.min(feature_preserve)
        feature_preserve_max = np.max(feature_preserve)
        if feature_preserve_max > feature_preserve_min:
            feature_preserve_normalized = (feature_preserve - feature_preserve_min) / (
                    feature_preserve_max - feature_preserve_min)
        else:
            feature_preserve_normalized = np.ones_like(feature_preserve)  # 如果所有值相同，则归一化为 1

        # 历史选择惩罚项
        selection_penalty = 1 / (np.sqrt(self.selected_counts) + 1)

        # 根据类别比例动态调整权重
        class_ratio = counts[minority_class] / counts[majority_class]
        w1 = 1 - self.resampler_alpha
        w2 = self.resampler_alpha

        # 组合得分（加入惩罚项）
        combined_scores = (w1 * (1 - boundary_sensitivity) + w2 * feature_preserve) * selection_penalty

        # 自适应概率校准
        quantiles = np.quantile(combined_scores, [0.25, 0.75])
        iqr = quantiles[1] - quantiles[0]
        scaled_scores = (combined_scores - quantiles[0]) / (iqr + 1e-8)
        selection_probs = np.clip(scaled_scores, 0, 1)
        selection_probs /= selection_probs.sum()

        # 重要性采样
        n_minority = counts[minority_class]
        selected_idx = np.random.choice(len(majority_samples),
                                        size=n_minority,
                                        p=selection_probs,
                                        replace=False)
        selected_indices = self.majority_indices_[selected_idx]

        return (np.vstack([majority_samples[selected_idx], minority_samples]),
                np.concatenate([np.full(n_minority, majority_class),
                                np.full(len(minority_samples), minority_class)]),
                selected_indices)

    def _process_subset(self, X_train, y_train):
        """处理子集：采样、训练分类器"""
        X_res,y_res, selected_indices=self.resampler(X_train,y_train)
        # 克隆分类器以确保独立性
        clfr = clone(self.clf)
        clfr.fit(X_res, y_res)
        return {
            "sub_data":X_res,
            "classifier": clfr,
        },X_res,y_res, selected_indices

    def _build_subset_models(self, X_low, y_train):
        # 初始化选择计数器
        counts = np.bincount(y_train)
        majority_class = np.argmax(counts)
        self.majority_indices_ = np.where(y_train == majority_class)[0]
        self.selected_counts = np.zeros(len(self.majority_indices_), dtype=int)

        # 模型构建流程
        all_candidates = []  # 存储所有候选子集的模型信息
        all_X_subs = []  # 存储所有候选子集的X_res
        all_y_subs = []  # 存储所有候选子集的y_res
        indices_candidates = []  # 存储所有候选子集的多数类样本索引
        n_candidates = self.sub_num
        i ,error_num,error_line=0,0,0.45
        while i < self.sub_num:
            i+=1
            # 生成子模型
            model_info, X_sub, y_sub, selected_indices = self._process_subset(X_low, y_train)
            sample_weights = np.ones(len(y_sub)) / len(y_sub)
            # 预测并计算错误
            y_pred = model_info['classifier'].predict(X_sub)
            errors = (y_pred != y_sub)
            error_rate = np.sum(errors * sample_weights) / np.sum(sample_weights)
            if error_rate >= error_line:  # 修改判断条件
                i = i - 1
                error_num+=1
                # print(error_num)
                if error_num % 100==0:
                    error_line+=0.01
                continue  # 跳过无效模型
            model_info['ACC']=1-error_rate
            all_candidates.append(model_info)
            # 更新选择计数器
            mask = np.isin(self.majority_indices_, selected_indices)
            self.selected_counts[mask] += 1
            # 存储候选模型
            all_X_subs.append(X_sub)
            all_y_subs.append(y_sub)
            indices_candidates.append(selected_indices)
        n_candidates = len(indices_candidates)
        # 计算多样性权重矩阵
        weight_diversity = np.zeros((n_candidates, n_candidates))
        for p in range(n_candidates):
            for q in range(n_candidates):
                intersection = len(set(indices_candidates[p]) & set(indices_candidates[q]))
                union = len(set(indices_candidates[p]) | set(indices_candidates[q]))
                weight_diversity[p, q] = 1 - intersection / union
        weight_diversity = (weight_diversity - np.min(weight_diversity)) / (
                    np.max(weight_diversity) - np.min(weight_diversity))

        majority_class = np.argmax(np.bincount(y_train))
        X_majority = X_low[y_train == majority_class]

        # 计算整体的均值和标准差（每个特征维度）
        mean_all = np.mean(X_majority, axis=0)
        std_all = np.std(X_majority, ddof=1, axis=0)

        # 合并为一个统计特征向量：[mean_1, mean_2, ..., mean_r, std_1, std_2, ..., std_r]
        L = np.concatenate([mean_all, std_all])
        r = len(mean_all)  # 特征维度数

        # 初始化相似性得分数组
        similarity = np.zeros(n_candidates)

        for p in range(n_candidates):
            subset_data = X_low[indices_candidates[p]]
            mean_p = np.mean(subset_data, axis=0)
            std_p = np.std(subset_data, ddof=1, axis=0)

            # 合并为当前子集的统计特征向量
            L_p = np.concatenate([mean_p, std_p])

            # 计算每维的绝对误差之和，并取平均
            avg_diff = np.sum(np.abs(L_p - L)) / (2 * r)

            # 应用 min 截断 + 转换为相似性得分
            similarity[p] = 1 - min(1.0, avg_diff)

        # 构造相似性权重矩阵 W^S，使用平均得分组合方式
        weight_similarity = np.zeros((n_candidates, n_candidates))
        for p in range(n_candidates):
            for q in range(n_candidates):
                weight_similarity[p, q] = np.sqrt(similarity[p] * similarity[q])


        # 综合权重矩阵（动态调整alpha）
        beta  = 0.5
        matrix = beta  * weight_diversity + (1 - beta ) * weight_similarity

        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

        # 计算每个候选子集的综合得分（取矩阵行均值）
        scores = np.mean(matrix, axis=1)

        # 筛选TOP30%的子集
        chose_num = int(self.sub_num * 0.3)
        top_indices = np.argsort(scores)[-chose_num:][::-1]  # 取得分最高的30%，按得分降序

        # 从 all_candidates 中取出对应模型，并为其添加 subset_score 字段
        subset_models = []
        for i in top_indices:
            model = all_candidates[i].copy()  # 建议 copy 避免修改原数据
            model["subset_score"] = scores[i]  # 直接挂载 subset-level score
            subset_models.append(model)

        # 收集各模型的训练预测结果
        train_preds_list = []
        for model in subset_models:
            clf = model["classifier"]  # 仍然可以直接使用 model["classifier"]
            pred_probs = clf.predict_proba(X_low)[:, 1]
            pred_labels = (pred_probs >= 0.5).astype(int)
            train_preds_list.append(pred_labels)

        return subset_models

    def _weighted_predict(self, models, X_test_low):
        """根据样本距离加权预测"""
        self.models=models
        y_pred = []
        for x_low in X_test_low:
            weights = []
            probs = []
            for model in models:
                sub_data = model["sub_data"]
                distances = np.linalg.norm(sub_data - x_low, axis=1)  # 欧氏距离
                # 剔除最远的样本
                num_samples_to_keep = int(len(distances) * 0.8)  # 保留stay_ratio的样本
                sorted_indices = np.argsort(distances)  # 按距离从小到大排序
                kept_indices = sorted_indices[:num_samples_to_keep]  # 取前stay_ratio
                kept_distances = distances[kept_indices]  # 保留的距离
                avg_distance = np.mean(kept_distances)
                # 归一化处理
                distance_weight = 1 / (1 + avg_distance) * model['ACC']
                # 模型预测概率
                prob = model["classifier"].predict_proba(x_low.reshape(1, -1))[0][1]
                weights.append(distance_weight)
                # weights.append(combined_weight)
                probs.append(prob)

                # 归一化权重
            weights = np.array(weights)
            if weights.sum() > 0:
                weights /= weights.sum()
            else:
                weights = np.ones_like(weights) / len(weights)

            final_prob = np.dot(weights, probs)
            y_pred.append(final_prob)

        return np.array(y_pred)  # 返回每个样本的正类概率

    def fit(self, X_train, y_train):
        self.pca = KernelPCA(**self.kernel_pca_params)
        # print(self.kernel_pca_params)
        X_train_low = self.pca.fit_transform(X_train)
        counts = np.bincount(y_train)
        majority_class = np.argmax(counts)
        minority_class = 1 - majority_class
        majority_count = counts[majority_class]
        minority_count = counts[minority_class]

        # 计算不平衡率
        IR = majority_count / minority_count
        # IR = 500
        if 200 >IR >= 20:
            # print("这个换成SVC训练")
            self.clf = SVC(kernel='rbf', probability=True, max_iter=10000)
        if IR >=200:
            # print("这个换成XGB训练")
            self.clf = XGBClassifier(
                learning_rate=0.1,
                n_estimators=200,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        subset_models = self._build_subset_models(X_train_low, y_train)
        return {
            'pca': self.pca,
            'subset_models': subset_models,
        }

    def predict(self, model_dict, X_test):
        """预测方法"""
        X_test_low = model_dict['pca'].transform(X_test)
        y_pred_probs = self._weighted_predict(model_dict['subset_models'], X_test_low)
        return np.where(y_pred_probs >= 0.5, 1, 0)

    def predict_proba(self,model_info, X_test):
        """预测概率"""
        X_test_low = model_info['pca'].transform(X_test)
        return self._weighted_predict(model_info['subset_models'], X_test_low)