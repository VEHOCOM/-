import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def test_model(model, test_dataset, batch_size, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn)
    
    accuracies = []
    
    for epoch in range(epochs):
        # 加载当前epoch的模型
        model.load_state_dict(torch.load(f'model_epoch_{epoch+1}.pth'))
        model.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for texts, labels, lengths in tqdm(dataloader, desc=f'Testing Epoch {epoch+1}'):
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts, lengths)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1} Test Accuracy: {accuracy:.4f}')
    
    return accuracies