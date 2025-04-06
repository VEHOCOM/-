import torch
import jieba
from model import BiLSTMModel
from dataset import WeiboDataset

def load_model(model_path, vocab_size):
    model = BiLSTMModel(
        vocab_size=vocab_size,
        embedding_dim=200,
        hidden_dim=256,
        output_dim=6,
        n_layers=2
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_emotion(text, model, vocab):
    # 情绪标签映射
    emotion_labels = {
        0: 'neutral（中性）',
        1: 'happy（开心）',
        2: 'angry（愤怒）',
        3: 'sad（悲伤）',
        4: 'fear（恐惧）',
        5: 'surprise（惊讶）'
    }
    
    # 文本预处理
    words = jieba.lcut(text)
    if len(words) == 0:
        words = ['<UNK>']
    
    # 转换为索引
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    text_tensor = torch.LongTensor(indices).unsqueeze(0)
    lengths = torch.LongTensor([len(indices)])
    
    # 使用模型预测
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    text_tensor = text_tensor.to(device)
    
    with torch.no_grad():
        output = model(text_tensor, lengths)
        prediction = torch.softmax(output, dim=1)
        pred_label = prediction.argmax(dim=1).item()
        confidence = prediction[0][pred_label].item()
    
    return emotion_labels[pred_label], confidence

def main():
    # 加载训练数据集以获取词表
    train_dataset = WeiboDataset("/Users/vehowang/Desktop/编着玩/双向LSTM微博文本情绪识别/评测数据集/train/usual_train.txt")
    
    # 加载第四轮的模型（使用完整路径）
    model = load_model('/Users/vehowang/Desktop/编着玩/双向LSTM微博文本情绪识别/model_epoch.pth', len(train_dataset.vocab))
    
    print("欢迎使用微博情绪分类系统！")
    print("输入 'q' 退出")
    
    while True:
        text = input("\n请输入要分析的文本: ")
        if text.lower() == 'q':
            break
            
        emotion, confidence = predict_emotion(text, model, train_dataset.vocab)
        print(f"\n预测情绪: {emotion}")
        print(f"置信度: {confidence:.2%}")

if __name__ == '__main__':
    main()