def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms
    # FashionMNIST 데이터셋 사용
    dataset = datasets.FashionMNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    # transforms.Compose는 방대한 데이터를 한번에 변환시켜주는 method
    # transforms.ToTensor()는 이미지를 torch.Tensor로 자동 변환
    # pixel 값을 0-255 -> 0.0-1.0으로 자동 변환

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y
