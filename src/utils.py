import torch


# set device (gpu or cpu)
def set_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    return device