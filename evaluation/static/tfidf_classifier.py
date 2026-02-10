"""
TF-IDF分类器模块

使用TF-IDF特征 + 传统机器学习分类器进行辅助诊断
"""

import json
import os
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import cross_val_score

try:
    from .config import (
        TFIDF_CONFIG, RANDOM_SEED, MODEL_DIR, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS
    )
    from .data_utils import (
        load_and_process_data, prepare_classification_dataset
    )
    from .metrics import (
        calculate_singlelabel_metrics, calculate_multilabel_metrics,
        format_metrics_for_print
    )
except ImportError:
    from config import (
        TFIDF_CONFIG, RANDOM_SEED, MODEL_DIR, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS
    )
    from data_utils import (
        load_and_process_data, prepare_classification_dataset
    )
    from metrics import (
        calculate_singlelabel_metrics, calculate_multilabel_metrics,
        format_metrics_for_print
    )


class TFIDFClassifier:
    """
    基于TF-IDF的分类器
    
    支持2分类、4分类和12分类（多标签）
    """
    
    def __init__(
        self,
        classification_type: str = "12class",
        classifier_type: str = "logistic",
        **kwargs
    ):
        """
        初始化分类器
        
        Args:
            classification_type: 分类类型 ("2class", "4class", "12class")
            classifier_type: 基础分类器类型 ("logistic", "svm", "rf")
        """
        self.classification_type = classification_type
        self.classifier_type = classifier_type
        
        # TF-IDF配置
        self.max_features = kwargs.get('max_features', TFIDF_CONFIG['max_features'])
        self.ngram_range = kwargs.get('ngram_range', TFIDF_CONFIG['ngram_range'])
        self.min_df = kwargs.get('min_df', TFIDF_CONFIG['min_df'])
        self.max_df = kwargs.get('max_df', TFIDF_CONFIG['max_df'])
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r'(?u)\b\w+\b'  # 支持中文单字符
        )
        
        # 初始化分类器
        self.classifier = self._create_classifier(classifier_type)
        
        # 标签编码器
        if classification_type == "12class":
            self.label_encoder = MultiLabelBinarizer(classes=TWELVE_CLASS_LABELS)
            self.is_multilabel = True
        else:
            self.label_encoder = LabelEncoder()
            self.is_multilabel = False
        
        # 标签列表
        if classification_type == "2class":
            self.labels = TWO_CLASS_LABELS
        elif classification_type == "4class":
            self.labels = FOUR_CLASS_LABELS
        else:
            self.labels = TWELVE_CLASS_LABELS
        
        self.is_fitted = False
    
    def _create_classifier(self, classifier_type: str):
        """创建基础分类器"""
        if classifier_type == "logistic":
            base_clf = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight='balanced',
                solver='lbfgs'
            )
        elif classifier_type == "svm":
            base_clf = LinearSVC(
                max_iter=2000,
                random_state=RANDOM_SEED,
                class_weight='balanced'
            )
        elif classifier_type == "rf":
            base_clf = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}")
        
        # 多标签分类使用OneVsRest
        if self.classification_type == "12class":
            return OneVsRestClassifier(base_clf)
        return base_clf
    
    def fit(self, texts: List[str], labels: List[Any]) -> 'TFIDFClassifier':
        """
        训练分类器
        
        Args:
            texts: 文本列表
            labels: 标签列表（单标签为字符串列表，多标签为列表的列表）
        
        Returns:
            self
        """
        print(f"正在训练TF-IDF {self.classification_type}分类器...")
        print(f"训练样本数: {len(texts)}")
        
        start_time = time.time()
        
        # TF-IDF特征提取
        X = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF特征维度: {X.shape}")
        
        # 标签编码
        if self.is_multilabel:
            y = self.label_encoder.fit_transform(labels)
        else:
            y = self.label_encoder.fit_transform(labels)
        
        # 训练分类器
        self.classifier.fit(X, y)
        
        self.is_fitted = True
        
        elapsed = time.time() - start_time
        print(f"训练完成，耗时: {elapsed:.2f}秒")
        
        return self
    
    def predict(self, texts: List[str]) -> List[Any]:
        """
        预测
        
        Args:
            texts: 文本列表
            
        Returns:
            预测标签列表
        """
        if not self.is_fitted:
            raise RuntimeError("分类器尚未训练，请先调用fit方法")
        
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        
        # 标签解码
        if self.is_multilabel:
            # 将二进制矩阵转换为标签列表
            predictions = []
            for row in y_pred:
                pred_labels = [self.labels[i] for i, v in enumerate(row) if v == 1]
                if not pred_labels:
                    pred_labels = ["Others"]
                predictions.append(pred_labels)
            return predictions
        else:
            return self.label_encoder.inverse_transform(y_pred).tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        预测概率
        
        Args:
            texts: 文本列表
            
        Returns:
            预测概率矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("分类器尚未训练，请先调用fit方法")
        
        X = self.vectorizer.transform(texts)
        
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        elif hasattr(self.classifier, 'decision_function'):
            # 对于SVM，使用decision_function
            return self.classifier.decision_function(X)
        else:
            raise RuntimeError("分类器不支持概率预测")
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[Any]
    ) -> Dict[str, Any]:
        """
        评估分类器
        
        Args:
            texts: 文本列表
            labels: 真实标签列表
            
        Returns:
            评估指标字典
        """
        predictions = self.predict(texts)
        
        if self.is_multilabel:
            metrics = calculate_multilabel_metrics(labels, predictions, self.labels)
        else:
            metrics = calculate_singlelabel_metrics(labels, predictions, self.labels)
        
        return metrics
    
    def save(self, path: str) -> None:
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'classification_type': self.classification_type,
                'classifier_type': self.classifier_type,
                'labels': self.labels,
                'is_multilabel': self.is_multilabel,
                'is_fitted': self.is_fitted
            }, f)
        print(f"模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TFIDFClassifier':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            classification_type=data['classification_type'],
            classifier_type=data['classifier_type']
        )
        instance.vectorizer = data['vectorizer']
        instance.classifier = data['classifier']
        instance.label_encoder = data['label_encoder']
        instance.labels = data['labels']
        instance.is_multilabel = data['is_multilabel']
        instance.is_fitted = data['is_fitted']
        
        return instance


def train_and_evaluate_tfidf(
    train_data: List[Dict],
    test_data: List[Dict],
    classification_type: str = "12class",
    classifier_type: str = "logistic",
    save_model: bool = True,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    训练并评估TF-IDF分类器
    
    Args:
        train_data: 训练数据
        test_data: 测试数据
        classification_type: 分类类型
        classifier_type: 分类器类型
        save_model: 是否保存模型
        sample_size: 训练样本数量（None表示使用全量数据）
        
    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    sample_info = f"样本数: {sample_size}" if sample_size else "全量样本"
    print(f"TF-IDF {classification_type}分类评估 (分类器: {classifier_type}, {sample_info})")
    print("="*60)
    
    # 准备数据
    train_texts, train_labels = prepare_classification_dataset(
        train_data, classification_type
    )
    test_texts, test_labels = prepare_classification_dataset(
        test_data, classification_type
    )
    
    # 如果指定了样本数，进行随机采样
    if sample_size is not None and sample_size < len(train_texts):
        print(f"原始训练集大小: {len(train_texts)}")
        print(f"随机采样到: {sample_size} 个样本")
        
        # 设置随机种子以确保可复现
        np.random.seed(RANDOM_SEED)
        sample_indices = np.random.choice(len(train_texts), sample_size, replace=False)
        
        train_texts = [train_texts[i] for i in sample_indices]
        if isinstance(train_labels[0], list):  # 多标签
            train_labels = [train_labels[i] for i in sample_indices]
        else:  # 单标签
            train_labels = [train_labels[i] for i in sample_indices]
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    # 创建并训练分类器
    clf = TFIDFClassifier(
        classification_type=classification_type,
        classifier_type=classifier_type
    )
    clf.fit(train_texts, train_labels)
    
    # 评估
    metrics = clf.evaluate(test_texts, test_labels)
    
    # 打印结果
    task_name = f"TF-IDF {classification_type}分类 ({classifier_type})"
    clf_type = "multi" if classification_type == "12class" else "single"
    print(format_metrics_for_print(metrics, task_name, clf_type))
    
    # 保存模型
    if save_model:
        model_path = os.path.join(
            MODEL_DIR, 
            f"tfidf_{classification_type}_{classifier_type}.pkl"
        )
        clf.save(model_path)
    
    return {
        'classification_type': classification_type,
        'classifier_type': classifier_type,
        'method': 'TF-IDF',
        'sample_size': sample_size if sample_size else len(train_texts),
        'metrics': metrics
    }


def run_tfidf_benchmark(
    train_file: str,
    test_file: str,
    classifier_types: List[str] = None
) -> List[Dict[str, Any]]:
    """
    运行完整的TF-IDF benchmark
    
    Args:
        train_file: 训练数据文件路径
        test_file: 测试数据文件路径
        classifier_types: 要测试的分类器类型列表
        
    Returns:
        所有评估结果列表
    """
    if classifier_types is None:
        classifier_types = ["logistic", "svm", "rf"]
    
    # 加载数据
    print("正在加载数据...")
    train_data = load_and_process_data(train_file)
    test_data = load_and_process_data(test_file)
    
    results = []
    
    # 对每种分类类型和分类器类型进行评估
    for clf_type in classifier_types:
        for class_type in ["2class", "4class", "12class"]:
            try:
                result = train_and_evaluate_tfidf(
                    train_data, test_data,
                    classification_type=class_type,
                    classifier_type=clf_type,
                    save_model=True
                )
                results.append(result)
            except Exception as e:
                print(f"评估 {class_type} ({clf_type}) 时出错: {e}")
                continue
    
    return results


if __name__ == "__main__":
    from .config import TRAIN_DATA_FILE, TEST_DATA_100_FILE
    
    # 运行benchmark
    results = run_tfidf_benchmark(
        TRAIN_DATA_FILE,
        TEST_DATA_100_FILE,
        classifier_types=["logistic"]  # 快速测试只用logistic
    )
    
    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, "tfidf_benchmark_results.json")
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

