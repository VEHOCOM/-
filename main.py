import argparse
import torch
import matplotlib.pyplot as plt
from dataset import WeiboDataset
from model import BiLSTMModel
from trainer import train_model
from tester import test_model

def main():
    parser = argparse.ArgumentParser(description='微博情绪分类训练程序')
    parser.add_argument('--train_path', type=str, required=True, help='训练集文件路径')
    parser.add_argument('--test_path', type=str, required=True, help='测试集文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    args = parser.parse_args()

    # 初始化数据集和模型
    train_dataset = WeiboDataset(args.train_path)
    test_dataset = WeiboDataset(args.test_path)
    
    model = BiLSTMModel(vocab_size=len(train_dataset.vocab),
                      embedding_dim=200,
                      hidden_dim=256,
                      output_dim=6,
                      n_layers=2)

    # 训练阶段
    train_losses = train_model(model, train_dataset, args.batch_size, args.epochs)
    
    # 测试阶段
    test_accuracies = test_model(model, test_dataset, args.batch_size, args.epochs)

    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == '__main__':
    main()