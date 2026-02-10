"""
BERT微调分类器模块

使用BERT进行精神疾病辅助诊断的文本分类
"""

import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers库未安装，BERT分类器将不可用")

try:
    from .config import (
        BERT_CONFIG, RANDOM_SEED, MODEL_DIR, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS
    )
    from .data_utils import load_and_process_data, prepare_classification_dataset
    from .metrics import (
        calculate_singlelabel_metrics, calculate_multilabel_metrics,
        format_metrics_for_print
    )
except ImportError:
    from config import (
        BERT_CONFIG, RANDOM_SEED, MODEL_DIR, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS
    )
    from data_utils import load_and_process_data, prepare_classification_dataset
    from metrics import (
        calculate_singlelabel_metrics, calculate_multilabel_metrics,
        format_metrics_for_print
    )


# 设置随机种子
def set_seed(seed: int = RANDOM_SEED):
    """设置随机种子以确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_long_text(
    text: str,
    tokenizer,
    max_length: int = 512,
    overlap: int = 128
) -> List[str]:
    """
    将长文本切分成多个片段
    
    Args:
        text: 原始文本
        tokenizer: tokenizer对象
        max_length: 每个片段的最大token数
        overlap: 相邻片段之间的重叠token数
    
    Returns:
        切分后的文本片段列表
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # 如果token数量不超过max_length-2（留出[CLS]和[SEP]的位置），直接返回原文本
    effective_max_length = max_length - 2  # 为[CLS]和[SEP]预留位置
    if len(tokens) <= effective_max_length:
        return [text]
    
    # 切分tokens
    chunks = []
    stride = effective_max_length - overlap
    
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + effective_max_length]
        # 将token ids转回文本
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # 如果这个chunk已经到达末尾，停止
        if i + effective_max_length >= len(tokens):
            break
    
    return chunks


def batch_split_long_texts(
    texts: List[str],
    labels: List[Any],
    tokenizer,
    max_length: int = 512,
    overlap: int = 128,
    batch_size: int = 1000
) -> Tuple[List[str], List[Any]]:
    """
    批量切分长文本（优化版本）
    
    Args:
        texts: 文本列表
        labels: 标签列表
        tokenizer: tokenizer对象
        max_length: 每个片段的最大token数
        overlap: 相邻片段之间的重叠token数
        batch_size: 批量编码的批次大小
    
    Returns:
        (切分后的文本列表, 对应的标签列表)
    """
    import logging
    # 临时抑制长度警告
    logger = logging.getLogger("transformers.tokenization_utils_base")
    original_level = logger.level
    logger.setLevel(logging.ERROR)
    
    effective_max_length = max_length - 2  # 为[CLS]和[SEP]预留位置
    stride = effective_max_length - overlap
    
    all_chunks = []
    all_labels = []
    
    # 批量处理
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Chunking Text"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # 批量编码
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            padding=False
        )
        
        # 处理每个文本
        for text, tokens, label in zip(batch_texts, encodings['input_ids'], batch_labels):
            if len(tokens) <= effective_max_length:
                # 不需要切分
                all_chunks.append(text)
                all_labels.append(label)
            else:
                # 需要切分
                for i in range(0, len(tokens), stride):
                    chunk_tokens = tokens[i:i + effective_max_length]
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    all_chunks.append(chunk_text)
                    all_labels.append(label)
                    
                    if i + effective_max_length >= len(tokens):
                        break
    
    # 恢复logging级别
    logger.setLevel(original_level)
    
    return all_chunks, all_labels


class TextClassificationDataset(Dataset):
    """文本分类数据集（支持长文本切分和随机采样）"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[Any],
        tokenizer,
        max_length: int = 512,
        is_multilabel: bool = False,
        label_to_id: Dict[str, int] = None,
        num_labels: int = None,
        enable_chunking: bool = False,
        overlap: int = 128,
        max_samples: int = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multilabel = is_multilabel
        self.label_to_id = label_to_id
        self.num_labels = num_labels
        self.enable_chunking = enable_chunking
        self.overlap = overlap
        self.max_samples = max_samples
        
        # 如果启用切分，则将长文本切分成多个片段（使用批量优化）
        if enable_chunking:
            self.texts, self.labels = batch_split_long_texts(
                texts, labels, tokenizer, max_length, overlap
            )
            
            # 如果设置了max_samples且切分后样本数超过限制，则随机采样
            if max_samples is not None and len(self.texts) > max_samples:
                indices = np.random.choice(len(self.texts), max_samples, replace=False)
                self.texts = [self.texts[i] for i in indices]
                self.labels = [self.labels[i] for i in indices]
        else:
            self.texts = list(texts)
            self.labels = list(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # 处理标签
        if self.is_multilabel:
            # 多标签：转换为多热编码
            label_vector = torch.zeros(self.num_labels)
            for l in label:
                if l in self.label_to_id:
                    label_vector[self.label_to_id[l]] = 1
            item['labels'] = label_vector
        else:
            # 单标签
            item['labels'] = torch.tensor(self.label_to_id[label])
        
        return item


class BERTClassifier(nn.Module):
    """BERT分类模型"""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        is_multilabel: bool = False
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.is_multilabel = is_multilabel
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.is_multilabel:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


class BERTTextClassifier:
    """
    BERT文本分类器封装类
    
    支持2分类、4分类和12分类（多标签）
    """
    
    def __init__(
        self,
        classification_type: str = "12class",
        model_name: str = None,
        max_length: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        num_epochs: int = None,
        device: str = None,
        freeze_bert: bool = True,
        enable_chunking: bool = True,
        chunk_overlap: int = 128,
        max_samples: int = None,
        early_stopping_patience: int = 3
    ):
        """
        初始化分类器
        
        Args:
            classification_type: 分类类型 ("2class", "4class", "12class")
            model_name: BERT模型名称
            max_length: 最大序列长度
            batch_size: 批次大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            device: 设备 ("cuda" 或 "cpu")
            freeze_bert: 是否冻结BERT层参数，只训练分类头 (默认: True)
            enable_chunking: 是否启用长文本切分 (默认: True)
            max_samples: 切分后随机采样的最大样本数 (默认: None, 不限制)
            early_stopping_patience: 早停耐心值，连续多少个epoch验证loss不下降则停止 (默认: 3)
            chunk_overlap: 切分时相邻片段的重叠token数 (默认: 128)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请先安装transformers库: pip install transformers")
        
        self.classification_type = classification_type
        self.model_name = model_name or BERT_CONFIG['model_name']
        self.max_length = max_length or BERT_CONFIG['max_length']
        self.batch_size = batch_size or BERT_CONFIG.get('batch_size', 512)
        self.learning_rate = learning_rate or BERT_CONFIG['learning_rate']
        self.num_epochs = num_epochs or BERT_CONFIG['num_epochs']
        self.freeze_bert = freeze_bert
        self.enable_chunking = enable_chunking
        self.chunk_overlap = chunk_overlap
        self.max_samples = max_samples if max_samples is not None else BERT_CONFIG.get('max_samples', None)
        self.early_stopping_patience = early_stopping_patience
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 标签设置
        if classification_type == "2class":
            self.labels = TWO_CLASS_LABELS
            self.is_multilabel = False
        elif classification_type == "4class":
            self.labels = FOUR_CLASS_LABELS
            self.is_multilabel = False
        else:  # 12class
            self.labels = TWELVE_CLASS_LABELS
            self.is_multilabel = True
        
        self.num_labels = len(self.labels)
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 模型将在fit时初始化
        self.model = None
        self.is_fitted = False
    
    def _create_dataloader(
        self,
        texts: List[str],
        labels: List[Any],
        shuffle: bool = True,
        enable_chunking: bool = None,
        max_samples: int = None
    ) -> DataLoader:
        """创建DataLoader
        
        Args:
            texts: 文本列表
            labels: 标签列表
            shuffle: 是否打乱数据
            enable_chunking: 是否启用切分（None则使用默认设置）
            max_samples: 切分后随机采样的最大样本数（None则不限制，仅对训练有效）
        """
        if enable_chunking is None:
            enable_chunking = self.enable_chunking
            
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            is_multilabel=self.is_multilabel,
            label_to_id=self.label_to_id,
            num_labels=self.num_labels,
            enable_chunking=enable_chunking,
            overlap=self.chunk_overlap,
            max_samples=max_samples
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def fit(
        self,
        train_texts: List[str],
        train_labels: List[Any],
        val_texts: List[str] = None,
        val_labels: List[Any] = None
    ) -> 'BERTTextClassifier':
        """
        训练分类器
        
        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            val_texts: 验证文本（可选）
            val_labels: 验证标签（可选）
        
        Returns:
            self
        """
        set_seed(RANDOM_SEED)
        
        print(f"\n正在训练BERT {self.classification_type}分类器...")
        print(f"模型: {self.model_name}")
        print(f"训练样本数: {len(train_texts)}")
        print(f"长文本切分: {'启用' if self.enable_chunking else '禁用'}")
        if self.enable_chunking:
            print(f"切分重叠: {self.chunk_overlap} tokens")
            if self.max_samples:
                print(f"最大采样数: {self.max_samples}")
        if val_texts:
            print(f"验证样本数: {len(val_texts)}")
        
        start_time = time.time()
        
        # 创建模型
        self.model = BERTClassifier(
            model_name=self.model_name,
            num_labels=self.num_labels,
            is_multilabel=self.is_multilabel
        )
        self.model.to(self.device)
        
        # 根据freeze_bert决定是否冻结BERT层参数
        if self.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False
            # 只有分类头的参数需要训练
            trainable_params = list(self.model.classifier.parameters()) + list(self.model.dropout.parameters())
            print(f"冻结BERT参数，只训练分类头")
        else:
            trainable_params = list(self.model.parameters())
            print(f"训练整个BERT模型")
        
        print(f"可训练参数数量: {sum(p.numel() for p in trainable_params if p.requires_grad)}")
        
        # 创建数据加载器（训练时启用切分和采样限制）
        train_dataloader = self._create_dataloader(
            train_texts, train_labels, shuffle=True, 
            enable_chunking=self.enable_chunking, max_samples=self.max_samples
        )
        print(f"切分后训练样本数: {len(train_dataloader.dataset)}" + 
              (f" (已采样，原限制: {self.max_samples})" if self.max_samples else ""))
        
        if val_texts:
            # 验证集不限制样本数
            val_dataloader = self._create_dataloader(
                val_texts, val_labels, shuffle=False, 
                enable_chunking=self.enable_chunking, max_samples=None
            )
            print(f"切分后验证样本数: {len(val_dataloader.dataset)}")
        
        # 优化器和调度器 - 只优化分类头参数
        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(train_dataloader) * self.num_epochs
        warmup_steps = int(total_steps * BERT_CONFIG.get('warmup_ratio', 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练循环
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_loss / len(train_dataloader)
            
            # 验证
            if val_texts:
                val_loss = self._evaluate_loss(val_dataloader)
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping: 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    # 深拷贝模型状态
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    print(f"  -> 新的最佳模型 (Val Loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  -> 验证loss未改善 ({patience_counter}/{self.early_stopping_patience})")
                    
                    if patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping! 验证loss连续{self.early_stopping_patience}个epoch未改善")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            print(f"\n已恢复到最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
        
        self.is_fitted = True
        
        elapsed = time.time() - start_time
        print(f"训练完成，耗时: {elapsed:.2f}秒")
        
        return self
    
    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        """计算验证集损失"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
        
        return total_loss / len(dataloader)
    
    def predict(self, texts: List[str]) -> List[Any]:
        """
        预测（支持长文本切分和soft vote）
        
        Args:
            texts: 文本列表
            
        Returns:
            预测标签列表
        """
        if not self.is_fitted:
            raise RuntimeError("分类器尚未训练，请先调用fit方法")
        
        self.model.eval()
        predictions = []
        
        # 逐个文本进行预测，如果启用切分则使用soft vote
        for text in texts:
            if self.enable_chunking:
                # 切分文本
                chunks = split_long_text(
                    text, self.tokenizer, self.max_length, self.chunk_overlap
                )
            else:
                chunks = [text]
            
            # 收集所有切片的概率
            all_probs = []
            
            for chunk in chunks:
                # 对单个切片进行编码
                encoding = self.tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs['logits']
                    
                    if self.is_multilabel:
                        probs = torch.sigmoid(logits)
                    else:
                        probs = torch.softmax(logits, dim=1)
                    
                    all_probs.append(probs.cpu().numpy())
            
            # Soft vote：对所有切片的概率取平均
            avg_probs = np.mean(all_probs, axis=0).squeeze()
            
            if self.is_multilabel:
                # 多标签：使用阈值判断
                pred = (avg_probs > 0.5)
                pred_labels = [self.id_to_label[i] for i, v in enumerate(pred) if v]
                if not pred_labels:
                    pred_labels = ["Others"]
                predictions.append(pred_labels)
            else:
                # 单标签：取概率最大的类别
                pred_idx = np.argmax(avg_probs)
                predictions.append(self.id_to_label[pred_idx])
        
        return predictions
    
    def predict_batch(self, texts: List[str]) -> List[Any]:
        """
        批量预测（不使用切分和soft vote，直接截断处理）
        
        Args:
            texts: 文本列表
            
        Returns:
            预测标签列表
        """
        if not self.is_fitted:
            raise RuntimeError("分类器尚未训练，请先调用fit方法")
        
        self.model.eval()
        predictions = []
        
        # 创建简单数据集（使用第一个有效标签作为占位符）
        first_label = self.labels[0]
        if self.is_multilabel:
            dummy_labels = [[first_label]] * len(texts)
        else:
            dummy_labels = [first_label] * len(texts)
        
        dataloader = self._create_dataloader(
            texts, dummy_labels, shuffle=False, enable_chunking=False
        )
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs['logits']
                
                if self.is_multilabel:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).cpu().numpy()
                    
                    for pred in preds:
                        pred_labels = [self.id_to_label[i] for i, v in enumerate(pred) if v]
                        if not pred_labels:
                            pred_labels = ["Others"]
                        predictions.append(pred_labels)
                else:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    predictions.extend([self.id_to_label[p] for p in preds])
        
        return predictions
    
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
        os.makedirs(path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(path)
        
        # 保存配置
        config = {
            'classification_type': self.classification_type,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'labels': self.labels,
            'is_multilabel': self.is_multilabel,
            'num_labels': self.num_labels,
            'label_to_id': self.label_to_id,
            'id_to_label': {str(k): v for k, v in self.id_to_label.items()},
            'enable_chunking': self.enable_chunking,
            'chunk_overlap': self.chunk_overlap,
            'max_samples': self.max_samples,
            'early_stopping_patience': self.early_stopping_patience
        }
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> 'BERTTextClassifier':
        """加载模型"""
        # 加载配置
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        instance = cls(
            classification_type=config['classification_type'],
            model_name=config['model_name'],
            max_length=config['max_length'],
            device=device,
            enable_chunking=config.get('enable_chunking', True),
            chunk_overlap=config.get('chunk_overlap', 128),
            max_samples=config.get('max_samples', None),
            early_stopping_patience=config.get('early_stopping_patience', 3)
        )
        
        instance.labels = config['labels']
        instance.is_multilabel = config['is_multilabel']
        instance.num_labels = config['num_labels']
        instance.label_to_id = config['label_to_id']
        instance.id_to_label = {int(k): v for k, v in config['id_to_label'].items()}
        
        # 加载tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # 加载模型
        instance.model = BERTClassifier(
            model_name=config['model_name'],
            num_labels=instance.num_labels,
            is_multilabel=instance.is_multilabel
        )
        instance.model.load_state_dict(
            torch.load(os.path.join(path, 'model.pt'), map_location=instance.device)
        )
        instance.model.to(instance.device)
        instance.is_fitted = True
        
        return instance


def train_and_evaluate_bert(
    train_data: List[Dict],
    test_data: List[Dict],
    classification_type: str = "12class",
    save_model: bool = True,
    val_ratio: float = 0.1,
    max_val_samples: int = 64,
    **kwargs
) -> Dict[str, Any]:
    """
    训练并评估BERT分类器
    
    Args:
        train_data: 训练数据
        test_data: 测试数据
        classification_type: 分类类型
        save_model: 是否保存模型
        val_ratio: 验证集比例 (默认: 0.1)
        
    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"BERT {classification_type}分类评估")
    print("="*60)
    
    # 准备数据
    all_train_texts, all_train_labels = prepare_classification_dataset(
        train_data, classification_type
    )
    test_texts, test_labels = prepare_classification_dataset(
        test_data, classification_type
    )
    
    # 切分训练集和验证集
    n_samples = len(all_train_texts)
    n_val = min(int(n_samples * val_ratio), max_val_samples)
    
    # 随机打乱索引
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_texts = [all_train_texts[i] for i in train_indices]
    train_labels = [all_train_labels[i] for i in train_indices]
    val_texts = [all_train_texts[i] for i in val_indices]
    val_labels = [all_train_labels[i] for i in val_indices]
    
    print(f"原始训练数据: {n_samples}")
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    # 创建并训练分类器
    clf = BERTTextClassifier(
        classification_type=classification_type,
        **kwargs
    )
    clf.fit(train_texts, train_labels, val_texts, val_labels)
    
    # 评估
    metrics = clf.evaluate(test_texts, test_labels)
    
    # 打印结果
    task_name = f"BERT {classification_type}分类"
    clf_type = "multi" if classification_type == "12class" else "single"
    print(format_metrics_for_print(metrics, task_name, clf_type))
    
    # 保存模型
    if save_model:
        model_path = os.path.join(MODEL_DIR, f"bert_{classification_type}")
        clf.save(model_path)
    
    return {
        'classification_type': classification_type,
        'method': 'BERT',
        'metrics': metrics
    }


def run_bert_benchmark(
    train_file: str,
    test_file: str,
    classification_types: List[str] = None
) -> List[Dict[str, Any]]:
    """
    运行完整的BERT benchmark
    
    Args:
        train_file: 训练数据文件路径
        test_file: 测试数据文件路径
        classification_types: 要测试的分类类型列表
        
    Returns:
        所有评估结果列表
    """
    if classification_types is None:
        classification_types = ["2class", "4class", "12class"]
    
    # 加载数据
    print("正在加载数据...")
    train_data = load_and_process_data(train_file)
    test_data = load_and_process_data(test_file)
    
    results = []
    
    for class_type in classification_types:
        try:
            result = train_and_evaluate_bert(
                train_data, test_data,
                classification_type=class_type,
                save_model=True
            )
            results.append(result)
        except Exception as e:
            print(f"评估 {class_type} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


if __name__ == "__main__":
    from .config import TRAIN_DATA_FILE, TEST_DATA_100_FILE
    
    if TRANSFORMERS_AVAILABLE:
        # 运行benchmark（只测试4分类作为快速验证）
        results = run_bert_benchmark(
            TRAIN_DATA_FILE,
            TEST_DATA_100_FILE,
            classification_types=["4class"]
        )
        
        # 保存结果
        output_file = os.path.join(OUTPUT_DIR, "bert_benchmark_results.json")
        
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
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
    else:
        print("请先安装transformers库: pip install transformers torch")

