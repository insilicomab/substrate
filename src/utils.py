import torch
from tqdm import tqdm


# set device (gpu or cpu)
def set_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    return device


def predict_classes(model, test_dataloader, device):
    preds = []
    for images, _ in tqdm(test_dataloader):
        images = images.to(device)
        
        model.eval()
        
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)
        pred = pred.to('cpu').numpy()

        preds.extend(pred)

    return preds


def predict(model, test_dataloader, device):
    outputs_list = []
    for images, _ in tqdm(test_dataloader):
        images = images.to(device)
        
        model.eval()
        
        outputs = model(images)
        outputs = outputs.to('cpu').detach().numpy()

        outputs_list.extend(outputs)

    return outputs_list