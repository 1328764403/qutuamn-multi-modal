"""
FCMR Task Switcher - Quick implementation for alternative tasks
快速切换 FCMR 数据集的不同任务定义
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class FCMRTaskSwitcher:
    """快速切换 FCMR 数据集的不同任务定义"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: FCMR 数据集的 DataFrame
        """
        self.df = df.copy()
    
    def get_labels(self, task: str, **kwargs) -> np.ndarray:
        """
        根据任务名称获取标签
        
        Args:
            task: 任务名称
                - 'original': 原始多标签分类（8维）
                - 'difficulty': 难度预测（3分类）
                - 'answer_count': 答案数量预测（4分类：0-3）
                - 'confidence': 置信度预测（回归）
                - 'answer_type': 答案类型分类（3分类：无/单/多）
                - 'anomaly': 异常检测（二分类）
            **kwargs: 任务特定参数
        Returns:
            labels: numpy array
        """
        if task == 'original':
            return self._encode_original_multilabel()
        elif task == 'difficulty':
            return self._encode_difficulty()
        elif task == 'answer_count':
            return self._encode_answer_count()
        elif task == 'confidence':
            return self._encode_confidence(**kwargs)
        elif task == 'answer_type':
            return self._encode_answer_type()
        elif task == 'anomaly':
            return self._encode_anomaly(**kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _encode_original_multilabel(self) -> np.ndarray:
        """原始多标签编码（8维）"""
        labels = []
        for _, row in self.df.iterrows():
            answer = self._get_answer(row)
            label = self._encode_answer(answer)
            labels.append(label)
        return np.array(labels, dtype=np.float32)
    
    def _encode_difficulty(self) -> np.ndarray:
        """难度预测：Easy=0, Medium=1, Hard=2"""
        difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        labels = []
        for _, row in self.df.iterrows():
            diff = str(row.get('difficulty', 'medium')).lower()
            label = difficulty_map.get(diff, 1)  # 默认 medium
            labels.append(label)
        return np.array(labels, dtype=np.int64)
    
    def _encode_answer_count(self) -> np.ndarray:
        """答案数量预测：0-3个答案"""
        labels = []
        for _, row in self.df.iterrows():
            answer = self._get_answer(row)
            if pd.isna(answer) or str(answer).strip() == 'None':
                count = 0
            else:
                count = len(str(answer).replace(' ', '').split(','))
            labels.append(min(count, 3))  # 限制在0-3
        return np.array(labels, dtype=np.int64)
    
    def _encode_confidence(self, base_conf_easy=0.9, base_conf_medium=0.6, 
                          base_conf_hard=0.3, penalty_per_answer=0.1) -> np.ndarray:
        """
        置信度预测（回归任务）
        
        Args:
            base_conf_easy: Easy题目的基础置信度
            base_conf_medium: Medium题目的基础置信度
            base_conf_hard: Hard题目的基础置信度
            penalty_per_answer: 每个答案的惩罚值
        """
        difficulty_map = {'easy': base_conf_easy, 'medium': base_conf_medium, 'hard': base_conf_hard}
        labels = []
        for _, row in self.df.iterrows():
            diff = str(row.get('difficulty', 'medium')).lower()
            base_conf = difficulty_map.get(diff, base_conf_medium)
            
            answer = self._get_answer(row)
            if pd.isna(answer) or str(answer).strip() == 'None':
                answer_count = 0
            else:
                answer_count = len(str(answer).replace(' ', '').split(','))
            
            confidence = max(0.1, base_conf - answer_count * penalty_per_answer)
            labels.append(confidence)
        return np.array(labels, dtype=np.float32)
    
    def _encode_answer_type(self) -> np.ndarray:
        """答案类型分类：0=无答案, 1=单选项, 2=多选项"""
        labels = []
        for _, row in self.df.iterrows():
            answer = self._get_answer(row)
            if pd.isna(answer) or str(answer).strip() == 'None':
                answer_type = 0  # 无答案
            else:
                answer_list = str(answer).replace(' ', '').split(',')
                if len(answer_list) == 1:
                    answer_type = 1  # 单选项
                else:
                    answer_type = 2  # 多选项
            labels.append(answer_type)
        return np.array(labels, dtype=np.int64)
    
    def _encode_anomaly(self, rare_threshold_percentile=10) -> np.ndarray:
        """
        异常检测：基于答案频率
        
        Args:
            rare_threshold_percentile: 罕见答案的百分位数阈值
        """
        # 统计答案频率
        answer_counts = self.df['answer'].value_counts()
        if 'correct_answer' in self.df.columns:
            answer_counts = self.df['correct_answer'].value_counts()
        
        # 计算阈值
        threshold = answer_counts.quantile(rare_threshold_percentile / 100.0)
        
        labels = []
        for _, row in self.df.iterrows():
            answer = self._get_answer(row)
            answer_str = str(answer) if not pd.isna(answer) else 'None'
            count = answer_counts.get(answer_str, 0)
            is_anomaly = 1 if count < threshold else 0
            labels.append(is_anomaly)
        return np.array(labels, dtype=np.int64)
    
    def _get_answer(self, row: pd.Series) -> Optional[str]:
        """从行中提取答案"""
        if 'answer' in row and pd.notna(row['answer']):
            return str(row['answer']).strip()
        elif 'correct_answer' in row and pd.notna(row['correct_answer']):
            return str(row['correct_answer']).strip()
        elif 'label' in row and pd.notna(row['label']):
            return str(row['label']).strip()
        return None
    
    def _encode_answer(self, answer: Optional[str]) -> np.ndarray:
        """
        将答案编码为8维多标签向量
        答案格式: "1", "2, 3", "1,2,3", "None"
        """
        label = np.zeros(8, dtype=np.float32)
        if answer is None or pd.isna(answer) or str(answer).strip() == 'None':
            return label
        
        # 解析答案
        answer_str = str(answer).replace(' ', '').strip()
        if answer_str == '' or answer_str == 'None':
            return label
        
        # 答案映射：None=0, 1=1, 2=2, 3=3, 1,2=4, 1,3=5, 2,3=6, 1,2,3=7
        answer_map = {
            'None': 0,
            '1': 1, '2': 2, '3': 3,
            '1,2': 4, '2,1': 4,
            '1,3': 5, '3,1': 5,
            '2,3': 6, '3,2': 6,
            '1,2,3': 7, '1,3,2': 7, '2,1,3': 7,
            '2,3,1': 7, '3,1,2': 7, '3,2,1': 7
        }
        
        answer_idx = answer_map.get(answer_str, 0)
        if answer_idx > 0:
            label[answer_idx] = 1.0
        
        return label
    
    def get_task_info(self, task: str) -> Dict:
        """
        获取任务信息（输出维度、任务类型等）
        
        Args:
            task: 任务名称
        Returns:
            dict: 任务信息
        """
        task_info = {
            'original': {
                'output_dim': 8,
                'task_type': 'classification',
                'is_multilabel': True,
                'description': '原始多标签分类（8个答案组合）'
            },
            'difficulty': {
                'output_dim': 3,
                'task_type': 'classification',
                'is_multilabel': False,
                'description': '难度预测（Easy/Medium/Hard）'
            },
            'answer_count': {
                'output_dim': 4,
                'task_type': 'classification',
                'is_multilabel': False,
                'description': '答案数量预测（0-3个答案）'
            },
            'confidence': {
                'output_dim': 1,
                'task_type': 'regression',
                'is_multilabel': False,
                'description': '置信度预测（回归任务）'
            },
            'answer_type': {
                'output_dim': 3,
                'task_type': 'classification',
                'is_multilabel': False,
                'description': '答案类型分类（无/单/多选项）'
            },
            'anomaly': {
                'output_dim': 1,
                'task_type': 'classification',
                'is_multilabel': False,
                'description': '异常检测（二分类）'
            }
        }
        
        if task not in task_info:
            raise ValueError(f"Unknown task: {task}")
        
        return task_info[task]


# Module-level helper (so train.py can query info without a DataFrame)
FCMR_TASK_INFO: Dict[str, Dict] = {
    "original": {
        "output_dim": 8,
        "task_type": "classification",
        "is_multilabel": True,
        "description": "原始多标签分类（8个答案组合）",
    },
    "difficulty": {
        "output_dim": 3,
        "task_type": "classification",
        "is_multilabel": False,
        "description": "难度预测（Easy/Medium/Hard）",
    },
    "answer_count": {
        "output_dim": 4,
        "task_type": "classification",
        "is_multilabel": False,
        "description": "答案数量预测（0-3个答案）",
    },
    "confidence": {
        "output_dim": 1,
        "task_type": "regression",
        "is_multilabel": False,
        "description": "置信度预测（回归任务）",
    },
    "answer_type": {
        "output_dim": 3,
        "task_type": "classification",
        "is_multilabel": False,
        "description": "答案类型分类（无/单/多选项）",
    },
    "anomaly": {
        "output_dim": 1,
        "task_type": "classification",
        "is_multilabel": False,
        "description": "异常检测（二分类）",
    },
}


def get_fcmr_task_info(task: str) -> Dict:
    """Get task info without constructing a switcher."""
    if task not in FCMR_TASK_INFO:
        raise ValueError(f"Unknown task: {task}")
    return FCMR_TASK_INFO[task]


# 使用示例
if __name__ == '__main__':
    # 示例：加载数据并切换任务
    import pandas as pd
    
    # 假设你已经加载了 FCMR 数据
    # df = pd.read_csv('data/fcmr/dataset/easy/easy_data.csv')
    
    # 创建任务切换器
    # switcher = FCMRTaskSwitcher(df)
    
    # 获取不同任务的标签
    # difficulty_labels = switcher.get_labels('difficulty')
    # answer_count_labels = switcher.get_labels('answer_count')
    # confidence_labels = switcher.get_labels('confidence')
    
    # 获取任务信息
    # task_info = switcher.get_task_info('difficulty')
    # print(f"Task: {task_info['description']}")
    # print(f"Output dim: {task_info['output_dim']}")
    # print(f"Task type: {task_info['task_type']}")
    
    print("FCMR Task Switcher ready!")
    print("Available tasks: original, difficulty, answer_count, confidence, answer_type, anomaly")
