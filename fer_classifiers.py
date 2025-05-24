import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FERClassifier:
    """
    Facial Expression Recognition Classifier using Classical Statistical Methods
    包含SVM、集成方法和多分類器融合策略
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm_classifier = None
        self.rf_classifier = None
        self.gb_classifier = None
        self.ensemble_classifier = None
        
        # 情感類別標籤 (基於FER 2013)
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
    
    def initialize_classifiers(self):
        """初始化所有分類器"""
        # SVM with optimized kernel functions
        self.svm_classifier = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Random Forest
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Gradient Boosting
        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Multi-classifier fusion (Voting)
        self.ensemble_classifier = VotingClassifier(
            estimators=[
                ('svm', self.svm_classifier),
                ('rf', self.rf_classifier),
                ('gb', self.gb_classifier)
            ],
            voting='soft'  # 使用機率進行投票
        )
    
    def optimize_hyperparameters(self, X_train, y_train):
        """超參數優化"""
        print("正在進行超參數優化...")
        
        # SVM超參數網格
        svm_param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # Random Forest超參數網格
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Gradient Boosting超參數網格
        gb_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        # SVM優化
        svm_grid = GridSearchCV(
            SVC(probability=True, random_state=42), 
            svm_param_grid, 
            cv=3, 
            scoring='accuracy',
            n_jobs=-1
        )
        svm_grid.fit(X_train, y_train)
        self.svm_classifier = svm_grid.best_estimator_
        print(f"最佳SVM參數: {svm_grid.best_params_}")
        
        # Random Forest優化
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42), 
            rf_param_grid, 
            cv=3, 
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        self.rf_classifier = rf_grid.best_estimator_
        print(f"最佳Random Forest參數: {rf_grid.best_params_}")
        
        # Gradient Boosting優化
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42), 
            gb_param_grid, 
            cv=3, 
            scoring='accuracy',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        self.gb_classifier = gb_grid.best_estimator_
        print(f"最佳Gradient Boosting參數: {gb_grid.best_params_}")
        
        # 更新集成分類器
        self.ensemble_classifier = VotingClassifier(
            estimators=[
                ('svm', self.svm_classifier),
                ('rf', self.rf_classifier),
                ('gb', self.gb_classifier)
            ],
            voting='soft'
        )
    
    def train(self, X_train, y_train, optimize=True):
        """訓練所有分類器"""
        print("開始訓練分類器...")
        
        # 特徵標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 初始化分類器
        self.initialize_classifiers()
        
        # 超參數優化（可選）
        if optimize:
            self.optimize_hyperparameters(X_train_scaled, y_train)
        
        # 訓練所有分類器
        print("訓練SVM...")
        self.svm_classifier.fit(X_train_scaled, y_train)
        
        print("訓練Random Forest...")
        self.rf_classifier.fit(X_train_scaled, y_train)
        
        print("訓練Gradient Boosting...")
        self.gb_classifier.fit(X_train_scaled, y_train)
        
        print("訓練集成分類器...")
        self.ensemble_classifier.fit(X_train_scaled, y_train)
        
        print("所有分類器訓練完成!")
    
    def predict(self, X_test):
        """使用所有分類器進行預測"""
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = {}
        predictions['SVM'] = self.svm_classifier.predict(X_test_scaled)
        predictions['Random Forest'] = self.rf_classifier.predict(X_test_scaled)
        predictions['Gradient Boosting'] = self.gb_classifier.predict(X_test_scaled)
        predictions['Ensemble'] = self.ensemble_classifier.predict(X_test_scaled)
        
        return predictions
    
    def predict_proba(self, X_test):
        """獲取預測機率"""
        X_test_scaled = self.scaler.transform(X_test)
        
        probabilities = {}
        probabilities['SVM'] = self.svm_classifier.predict_proba(X_test_scaled)
        probabilities['Random Forest'] = self.rf_classifier.predict_proba(X_test_scaled)
        probabilities['Gradient Boosting'] = self.gb_classifier.predict_proba(X_test_scaled)
        probabilities['Ensemble'] = self.ensemble_classifier.predict_proba(X_test_scaled)
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """評估所有分類器性能"""
        predictions = self.predict(X_test)
        results = {}
        
        for classifier_name, y_pred in predictions.items():
            results[classifier_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """交叉驗證"""
        X_scaled = self.scaler.fit_transform(X)
        
        cv_results = {}
        classifiers = {
            'SVM': self.svm_classifier,
            'Random Forest': self.rf_classifier,
            'Gradient Boosting': self.gb_classifier,
            'Ensemble': self.ensemble_classifier
        }
        
        for name, classifier in classifiers.items():
            scores = cross_val_score(classifier, X_scaled, y, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores
            }
        
        return cv_results
    
    def plot_confusion_matrix(self, X_test, y_test, classifier_name='Ensemble'):
        """繪製混淆矩陣"""
        predictions = self.predict(X_test)
        y_pred = predictions[classifier_name]
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.emotion_labels.values()),
                   yticklabels=list(self.emotion_labels.values()))
        plt.title(f'Confusion Matrix - {classifier_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_performance_comparison(self, results):
        """繪製性能比較圖"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        classifiers = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[clf][metric] for clf in classifiers]
            bars = axes[i].bar(classifiers, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            
            # 在柱狀圖上添加數值標籤
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """特徵重要性分析（僅適用於Random Forest和Gradient Boosting）"""
        importance_results = {}
        
        if hasattr(self.rf_classifier, 'feature_importances_'):
            importance_results['Random Forest'] = self.rf_classifier.feature_importances_
        
        if hasattr(self.gb_classifier, 'feature_importances_'):
            importance_results['Gradient Boosting'] = self.gb_classifier.feature_importances_
        
        return importance_results
    
    def generate_classification_report(self, X_test, y_test):
        """生成詳細的分類報告"""
        predictions = self.predict(X_test)
        
        for classifier_name, y_pred in predictions.items():
            print(f"\n{'='*50}")
            print(f"Classification Report - {classifier_name}")
            print(f"{'='*50}")
            print(classification_report(y_test, y_pred, 
                                      target_names=list(self.emotion_labels.values())))

# 使用範例
def demo_fer_classifier():
    """演示FER分類器的使用"""
    # 生成模擬數據（實際使用時替換為真實的特徵數據）
    np.random.seed(42)
    n_samples = 1000
    n_features = 100  # 假設有100個特徵（LBP + HOG + Gabor特徵組合）
    n_classes = 7     # 7種情感
    
    # 模擬特徵數據
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 創建FER分類器
    fer_classifier = FERClassifier()
    
    # 訓練分類器
    fer_classifier.train(X_train, y_train, optimize=False)  # 設為False以加快演示速度
    
    # 評估性能
    results = fer_classifier.evaluate(X_test, y_test)
    
    # 顯示結果
    print("\n分類器性能評估結果:")
    print("="*60)
    for classifier_name, metrics in results.items():
        print(f"\n{classifier_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # 繪製性能比較圖
    fer_classifier.plot_performance_comparison(results)
    
    # 生成詳細分類報告
    fer_classifier.generate_classification_report(X_test, y_test)
    
    # 繪製混淆矩陣
    fer_classifier.plot_confusion_matrix(X_test, y_test, 'Ensemble')
    
    # 交叉驗證
    cv_results = fer_classifier.cross_validate(X_train, y_train, cv=3)
    print("\n交叉驗證結果:")
    print("="*40)
    for classifier_name, cv_metrics in cv_results.items():
        print(f"{classifier_name}: {cv_metrics['mean_accuracy']:.4f} ± {cv_metrics['std_accuracy']:.4f}")

if __name__ == "__main__":
    demo_fer_classifier()
