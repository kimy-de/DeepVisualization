# import libraries
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
import PIL
import scipy.ndimage as nd
import PIL.Image
from tqdm import tqdm

def init_image(size=(400, 400, 3)):

    normalise = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = PIL.Image.fromarray(np.uint8(np.random.uniform(150, 180, size)))
    img_tensor = normalise(img).unsqueeze(0)
    img_np = img_tensor.numpy()
    return img_np

def tensor_to_img(t):
    a = t.numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255
    inp = np.uint8(np.clip(inp, 0, 255))
    return PIL.Image.fromarray(inp)

# Octaver function
def octaver_fn(model, base_img, step_fn, octave_n=6, octave_scale=1.4, iter_n=10, **step_args):
    octaves = [base_img]

    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]

        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src = octave_base + detail

        for i in range(iter_n):
            src = step_fn(model, src, **step_args)

        detail = src.numpy() - octave_base

    return src

# Filter visualization
def filter_step(model, img, layer_index, filter_index, step_size=3, use_L2=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    model.zero_grad()

    img_var = Variable(torch.Tensor(img).to(device), requires_grad=True)
    optimizer = Adam([img_var], lr=step_size, weight_decay=1e-4)

    x = img_var
    for index, layer in enumerate(model.features):

        x = layer(x)
        if index == layer_index:

            break

    output = x[0, filter_index]
    loss =  output.norm() #torch.mean(output)
    loss.backward()

    if use_L2 == True:
        # L2 normalization on gradients
        mean_square = torch.Tensor([torch.mean(img_var.grad.data ** 2) + 1e-5]).to(device)
        img_var.grad.data /= torch.sqrt(mean_square)
        img_var.data.add_(img_var.grad.data * step_size)

    else:
        optimizer.step()

    result = img_var.data.cpu().numpy()
    result[0, :, :, :] = np.clip(result[0, :, :, :], -mean / std, (1 - mean) / std)

    return torch.Tensor(result)

def visualize_filter(model, base_img, layer_index, filter_index,
                     octave_n=6, octave_scale=1.4, iter_n=10,
                     step_size=3, use_L2=False):
    return octaver_fn(
        model, base_img, step_fn=filter_step,
        octave_n=octave_n, octave_scale=octave_scale,
        iter_n=iter_n, layer_index=layer_index,
        filter_index=filter_index, step_size=step_size,
        use_L2=use_L2
    )

# Show
def show_layer(model, layer_num, img_size, save_path, filter_start=10, filter_end=20, step_size=3, use_L2=False):

    img_np = init_image(size=(img_size, img_size, 3))

    for i in tqdm(range(filter_start, filter_end), desc="Feature Visualization", mininterval=0.01):
        title = "Layer {} Filter {}".format(layer_num, i)

        filter = visualize_filter(model, img_np, layer_num,
                                  filter_index=i,
                                  octave_n=2,
                                  iter_n=20,
                                  step_size=step_size,
                                  use_L2=use_L2)

        filter_img = tensor_to_img(filter)
        filter_img.save(save_path + title + ".png")