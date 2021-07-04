import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models import resnet50
from PIL import Image
import os
import numpy as np

net = resnet50(pretrained=True)
modules = list(net.children())[:-1]
feature_extractor = torch.nn.Sequential(*modules)
for p in feature_extractor.parameters():
    p.requires_grad = False

centre_crop = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['minivan', 'porcupine_hedgehog']
path = {'porcupine_hedgehog': '../../n02346627', 'minivan': '../../n03770679'}
images = {}
features = {c: [] for c in classes}
map_target = {c: i for i, c in enumerate(classes)}

for c in classes:
    images[c] = os.listdir(path[c])
    print(len(images[c]))

for c in classes:
    print(f"Starting class {c}")
    for i, im in enumerate(images[c]):
        path_img = os.path.join(path[c], im)
        img = Image.open(path_img)
        try:
            out = feature_extractor(Variable(centre_crop(img).unsqueeze(0)))
            features[c].append(np.append(out.numpy().flatten(), float(map_target[c])))
        except:
            print("Skipping image")
        if i % 100 == 0:
            print(i)

for c in classes:
    np.save(c + '.npy', np.array(features[c]))
    np.savetxt(c + '.csv', np.array(features[c]), delimiter=' ', fmt='%f')


