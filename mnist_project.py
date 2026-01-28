from pyexpat import model
import os, glob, random 
import numpy as np
from scipy import ndimage
import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")

def nll_from_softmax_probs(probs, targets, eps=1e-9):
    """
    probs: output of Softmax => shape [batch, 10]
    targets: true class indices => shape [batch]
    I'll compute log(probs) and apply NLLLoss.
    eps is added for numerical stability to avoid log(0).
    """
    log_probs = torch.log(probs + eps)
    return nn.NLLLoss()(log_probs, targets)

class MLP(nn.Module):
    """
    Supports:
    - activation: str ("sigmoid", "tanh", "relu") applied to all 6 hidden layers
    - activations: list of 6 strings for per-layer mixed activations
    - Output is ALWAYS Softmax.
    """
    def __init__(self, activation="sigmoid", activations=None, dropout=0.2):
        super().__init__()

        layer_sizes = [
            (784, 256),
            (256, 256),
            (256, 128),
            (128, 128),
            (128, 64),
            (64, 32),
        ]

        # Decide per-layer activations
        if activations is None:
            activations = [activation] * 6
        if len(activations) != 6:
            raise ValueError("activations must be a list of length 6 (one per hidden layer).")

        blocks = []
        for (in_f, out_f), act_name in zip(layer_sizes, activations):
            blocks.append(nn.Linear(in_f, out_f))
            blocks.append(get_activation(act_name))
            blocks.append(nn.Dropout(dropout))

        # Output layer (Softmax)
        blocks.append(nn.Linear(32, 10))
        blocks.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def get_mnist_dataloaders(batch_size=128):
    tfm = transforms.Compose([
        transforms.ToTensor(), # [0,1] range
        transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 images to 784 vector
    ])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tfm)
    return(
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )
    
def run_one_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            out = model(x)  # probabilities after Softmax
            loss = nll_from_softmax_probs(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def train_mnist(activation="sigmoid", activations=None, epochs=10, lr=1e-3, batch_size=128, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataloaders(batch_size)

    model = MLP(activation=activation, activations=activations, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, optimizer, device, train=True)
        te_loss, te_acc = run_one_epoch(model, test_loader, optimizer, device, train=False)

        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss); hist["test_acc"].append(te_acc)

        tag = "MIXED" if activations is not None else activation.upper()
        print(f"[{tag}] Epoch {ep:02d}/{epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | test loss={te_loss:.4f} acc={te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, hist, best_acc


def save_mnist_artifacts(hist, out_dir, title_prefix):
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["test_loss"], label="test")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend(); plt.title(f"{title_prefix} Loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["test_acc"], label="test")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.legend(); plt.title(f"{title_prefix} Accuracy")
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=200)
    plt.close()

def preprocess_custom_image(path, invert_auto=True):
    img = Image.open(path).convert("L")  # grayscale
    img = ImageOps.autocontrast(img) 

    if invert_auto and np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    img = img.filter(ImageFilter.MaxFilter(3)) 

    arr = np.array(img).astype(np.float32) / 255.0  # normalize to [0,1]
    mask = arr > 0.15

    labeled, n = ndimage.label(mask)
    if n == 0:
        arr28 = np.zeros((28, 28), dtype=np.float32)
        return arr28.reshape(1, -1), arr28, img
        
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = 1 + np.argmax(sizes)
    mask = (labeled == largest)
    arr = arr * mask

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]
   
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # resize to 28x28
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas.paste(cropped_img, (x_offset, y_offset))

    arr28 = np.array(canvas).astype(np.float32) / 255.0

    cy, cx = ndimage.center_of_mass(arr28)
    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))
    arr28 = ndimage.shift(arr28, shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)

    flat = arr28.reshape(1, -1)
    return flat, arr28, img

    
def predict_one(model, path, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        flat, img28, img_gray = preprocess_custom_image(path)

        x = torch.tensor(flat, dtype=torch.float32)

        device = next(model.parameters()).device
        x = x.to(device)

        model.eval()
        with torch.no_grad():
            probs = model(x)[0].detach().cpu().numpy()

        pred = int(np.argmax(probs))

        fname = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(out_dir, f"{fname}_prediction.png")

        plt.figure(figsize=(11, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(img_gray), cmap='gray')
        plt.title("Loaded (grayscale)"); plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img28, cmap='gray')
        plt.title("Preprocessed (28x28)"); plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.bar(range(10), probs)
        plt.xticks(range(10))
        plt.title(f"Predicted: {pred}")
        plt.xlabel("Digit"); plt.ylabel("Probability")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        return pred

def evaluate_folder(model, folder, out_dir):
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    if len(paths) == 0:
        print(f"WARNING: No images found in folder: {folder}")
        return 0.0

    y_true, y_pred = [], []

    for p in paths:
        base = os.path.basename(p)
        try:
            true_label = int(base.split('_')[0])
        except ValueError:
            print(f"WARNING: Skipping file with invalid name format: {base}")
            continue

        pred = predict_one(model, p, out_dir)
        y_true.append(true_label)
        y_pred.append(pred)

    acc = (np.array(y_true) == np.array(y_pred)).mean()

    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix: {os.path.basename(folder)}")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    return acc

def save_mnist_sample_grid(out_path= "outputs/mnist_sample_grid.png", n=9, seed=42):
    set_seed(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tfm = transforms.ToTensor()
    ds = datasets.MNIST("data", train=True, download=True, transform=tfm)

    idxs = np.random.choice(len(ds), size=n, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, idx in zip(axes.flatten(), idxs):
        img, label = ds[idx]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved MNIST sample grid: {out_path}")


def save_accuracy_summary(results_dict, out_path="outputs/accuracy_summary.csv"):
    """
    results_dict format:
    { model_name: { folder_name: accuracy_float, ... }, ... }
    Saves a CSV table.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Determine folder columns
    all_folders = sorted({folder for m in results_dict.values() for folder in m.keys()})
    model_names = sorted(results_dict.keys())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("model," + ",".join(all_folders) + "\n")
        for model in model_names:
            row = [model]
            for folder in all_folders:
                val = results_dict[model].get(folder, "")
                row.append(f"{val:.4f}" if isinstance(val, float) else "")
            f.write(",".join(row) + "\n")

    print(f"\nSaved accuracy summary CSV: {out_path}")

def save_mnist_best_summary(rows, out_path="outputs/mnist_best_summary.csv"):
    """
    rows: list of tuples [(model_name, best_acc), ...]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("model,best_mnist_test_accuracy\n")
        for name, acc in rows:
            f.write(f"{name},{acc:.4f}\n")
    print(f"Saved MNIST best summary CSV: {out_path}")


def main():
    epochs = 15
    batch_size = 128
    lr = 1e-3

    save_mnist_sample_grid("outputs/mnist_sample_grid.png")

    # =========================
    # BASELINE (REQUIRED)
    # =========================
    sig_model, sig_hist, sig_best = train_mnist("sigmoid", epochs=epochs, lr=lr, batch_size=batch_size)
    save_mnist_artifacts(sig_hist, "outputs/mnist_sigmoid", "MNIST Sigmoid")

    tanh_model, tanh_hist, tanh_best = train_mnist("tanh", epochs=epochs, lr=lr, batch_size=batch_size)
    save_mnist_artifacts(tanh_hist, "outputs/mnist_tanh", "MNIST Tanh")

    print("\nMNIST TEST ACCURACY (BEST) - BASELINE:")
    print(f" Sigmoid: {sig_best:.4f}")
    print(f" Tanh:    {tanh_best:.4f}")

    # =========================
    # ADDITIONAL EXPERIMENTS
    # =========================
    extra_experiments = {
        "relu_only": {"activation": "relu", "activations": None},
        "sigmoid_tanh_mix": {"activation": "sigmoid", "activations": ["sigmoid","sigmoid","sigmoid","tanh","tanh","tanh"]},
        "tanh_relu_mix": {"activation": "tanh", "activations": ["tanh","tanh","tanh","relu","relu","relu"]},
        "sigmoid_tanh_relu_mix": {"activation": "sigmoid", "activations": ["tanh","relu","sigmoid","tanh","relu","sigmoid"]},
    }

    extra_models = {}
    extra_results = []

    for name, cfg in extra_experiments.items():
        model, hist, best = train_mnist(
            activation=cfg["activation"],
            activations=cfg["activations"],
            epochs=epochs, lr=lr, batch_size=batch_size
        )
        save_mnist_artifacts(hist, f"outputs/mnist_{name}", f"MNIST {name}")
        extra_models[name] = model
        extra_results.append((name, best))

    print("\nADDITIONAL EXPERIMENTS (BEST MNIST TEST ACC):")
    for name, acc in extra_results:
        print(f" {name}: {acc:.4f}")

    mnist_rows = [("sigmoid", sig_best), ("tanh", tanh_best)] + extra_results
    save_mnist_best_summary(mnist_rows, out_path="outputs/mnist_best_summary.csv")

    # =========================
    # EXTERNAL / CUSTOM TESTING
    # =========================
    tests = {
        "word_blackbg": "custom_digits/word_blackbg",
        "word_whitebg": "custom_digits/word_whitebg",
        "paint_whitebg": "custom_digits/paint_whitebg",
        "online": "custom_digits/online",
        "handwritten": "custom_digits/handwritten"
    }

    models_to_test = {
        "sigmoid": sig_model,
        "tanh": tanh_model,
        # optionally test extra models too:
        **extra_models
    }

    print("\nCustom Folder accuracies (All Models):")

    accuracy_results = {}

    for model_name, model in models_to_test.items():
        accuracy_results[model_name] = {}

        print(f"\n--- Testing model: {model_name.upper()} ---")
        for folder_name, folder_path in tests.items():
            out_dir = os.path.join("outputs", "custom_tests", model_name, folder_name)
            acc = evaluate_folder(model, folder_path, out_dir)

            accuracy_results[model_name][folder_name] = acc
            print(f" Folder: {folder_name} | Accuracy: {acc:.4f}")

    save_accuracy_summary(accuracy_results, out_path="outputs/accuracy_summary.csv")
    



if __name__ == "__main__":
    main()