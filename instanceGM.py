# Reference:
# 1. DivideMix: https://github.com/LiJunnan1992/DivideMix
# 2. CausalNL: https://github.com/a5507203/IDLN
# Our code is heavily based on the above-mentioned repositories.

import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from mylib.models.vae import VAE_CIFAR10
from PreResNet import ResNet18

# Number of warm-up epochs
WARM_UP = 10
# Momenta
MOM1 = 0.9
MOM2 = 0.1


# Training
def train(
    epoch,
    net,
    net2,
    optimizer,
    labeled_trainloader,
    unlabeled_trainloader,
    train_loader,
    vae_model_1,
    vae_model_2,
    optimizer_vae,
    device,
    net_1=True,
):
    net.train()
    vae_model_1.train()
    vae_model_2.eval()
    net2.eval()  # fix one network and train the other
    criterion = SemiLoss()
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(
        labeled_trainloader
    ):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(
            1, labels_x.view(-1, 1), 1
        )
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = (
            inputs_x.cuda(),
            inputs_x2.cuda(),
            labels_x.cuda(),
            w_x.cuda(),
        )
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (
                torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)
            ) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # mixmatch
        lambda_mix = np.random.beta(args.alpha, args.alpha)
        lambda_mix = max(lambda_mix, 1 - lambda_mix)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = lambda_mix * input_a + (1 - lambda_mix) * input_b
        mixed_target = lambda_mix * target_a + (1 - lambda_mix) * target_b

        logits = net(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        Lx, Lu, lamb = criterion(
            logits_x,
            mixed_target[: batch_size * 2],
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch + batch_idx / num_iter,
            WARM_UP,
        )

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss_dm = Lx + lamb * Lu + penalty

        vae_args.alpha_plan = [vae_args.lr] * vae_args.EPOCHS
        vae_args.beta1_plan = [MOM1] * vae_args.EPOCHS

        for i in range(vae_args.epoch_decay_start, vae_args.EPOCHS):
            vae_args.alpha_plan[i] = (
                float(vae_args.EPOCHS - i)
                / (vae_args.EPOlambCHS - vae_args.epoch_decay_start)
                * vae_args.lr
            )
            vae_args.beta1_plan[i] = MOM2

        vae_args.rate_schedule = np.ones(vae_args.EPOCHS) * vae_args.forget_rate
        vae_args.rate_schedule[: vae_args.num_gradual] = np.linspace(
            0, vae_args.forget_rate**vae_args.exponent, vae_args.num_gradual
        )

        adjust_learning_rate(optimizer_vae, epoch)

        loss_vae, reconst_x, noisy_y_ce, uniform_x, gaussian_z = train_vae(
            train_loader, device, net, vae_model_1, optimizer_vae
        )

        loss = loss_dm + loss_vae

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_vae.step()

    sys.stdout.write("\r")
    sys.stdout.write(
        (
            "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t"
            " Labeled loss: %.2f  Unlabeled loss: %.2f"
        )
        % (
            args.dataset,
            args.r,
            args.noise_mode,
            epoch,
            args.num_epochs,
            batch_idx + 1,
            num_iter,
            Lx.item(),
            Lu.item(),
        )
    )
    sys.stdout.flush()
    return loss


# Train vae
def train_vae(train_loader, device, net, vae_model1, optimizer_vae):
    vae_model1.train()
    for _, (data, targets, _) in enumerate(train_loader):
        optimizer_vae.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        # forward
        x_hat1, n_logits1, mu1, log_var1, c_logits1, y_hat1 = vae_model1(data, net)
        x_hat1, n_logits1, mu1, log_var1, c_logits1, y_hat1 = (
            x_hat1.cuda(),
            n_logits1.cuda(),
            mu1.cuda(),
            log_var1.cuda(),
            c_logits1.cuda(),
            y_hat1.cuda(),
        )
        # calculate loss
        vae_loss_1, l1, l2, l3, l4 = vae_loss(
            x_hat1, data, n_logits1, targets, mu1, log_var1, c_logits1, y_hat1
        )
        return vae_loss_1, l1, l2, l3, l4


# two component GMM model
def eval_train(model, all_loss, eval_loader):
    model.eval()
    CE = nn.CrossEntropyLoss(reduction="none")
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for _, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)
    SMALL_DATASET = 1000
    if (
        len(eval_loader.dataset) < SMALL_DATASET or args.r == 0.9
    ):  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


# Testing
def test(epoch, net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.0 * correct / total
    logging.info("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))


# %%
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


# %%
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


# %%
class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


# %%
def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    CEloss = nn.CrossEntropyLoss()

    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        loss.backward()
        optimizer.step()
        sys.stdout.write("\r")
        sys.stdout.write(
            "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f"
            % (
                args.dataset,
                args.r,
                args.noise_mode,
                epoch,
                args.num_epochs,
                batch_idx + 1,
                num_iter,
                loss.item(),
            )
        )
        sys.stdout.flush()


# %%
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group["lr"] = vae_args.alpha_plan[epoch]
        param_group["betas"] = (vae_args.beta1_plan[epoch], 0.999)  # Only change beta1


def log_standard_categorical(p, reduction="mean"):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    if reduction == "mean":
        cross_entropy = torch.mean(cross_entropy)
    else:
        cross_entropy = torch.sum(cross_entropy)
    return cross_entropy


# VAE Loss
def vae_loss(x_hat, data, n_logits, targets, mu, log_var, c_logits, h_c_label):
    # x loss
    c_bernoulli = torch.distributions.continuous_bernoulli.ContinuousBernoulli(
        probs=x_hat
    )
    reconstruction_losses = -c_bernoulli.log_prob(value=data)  # (N, C, H, W)
    l1 = torch.mean(input=reconstruction_losses)  # scalar
    # \tilde{y]} loss
    l2 = F.cross_entropy(n_logits, targets, reduction="mean")
    #  uniform loss for x
    l3 = -0.00001 * log_standard_categorical(h_c_label, reduction="mean")
    #  Gaussian loss for z
    l4 = -0.0003 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (l1 + l2 + l3 + l4), l1, l2, l3, l4


args = types.SimpleNamespace()
vae_args = types.SimpleNamespace()


def main(loader, checkpoint_file, num_class, project_name):
    args.batch_size = 64
    args.lr = 0.002
    args.vae_lr = 0.001
    args.noise_mode = "instance"
    args.alpha = 4
    args.lambda_u = 25
    args.p_threshold = 0.5
    args.T = 0.5
    args.num_epochs = 25
    args.r = 0.5
    args.seed = 123
    args.gpuid = 0
    args.num_class = num_class
    args.dataset = project_name
    args.z_dim = 25

    # %%
    logging.info("| Building net")
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    # %%
    optimizer1 = optim.SGD(
        net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    optimizer2 = optim.SGD(
        net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )

    all_loss = [[], []]  # save the history of losses from two networks

    temp_ = loader.run("warmup")
    img, target, _ = next(iter(temp_))

    # %%
    vae_args.lr = 0.001
    vae_args.LOG_INTERVAL = 100
    vae_args.BATCH_SIZE = args.batch_size
    vae_args.EPOCHS = args.num_epochs + 1
    vae_args.z_dim = args.z_dim
    vae_args.dataset = args.dataset
    vae_args.select_ratio = 0.25
    vae_args.epoch_decay_start = 1000
    vae_args.noise_rate = args.r
    vae_args.forget_rate = args.r
    vae_args.exponent = 1
    vae_args.num_gradual = 10

    vae_model1 = VAE_CIFAR10(z_dim=args.z_dim, num_classes=args.num_class)
    vae_model2 = VAE_CIFAR10(z_dim=args.z_dim, num_classes=args.num_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = {"vae_model1": vae_model1.to(device), "vae_model2": vae_model2.to(device)}

    # %%
    optimizers = {
        "vae1": torch.optim.Adam(model["vae_model1"].parameters(), lr=args.vae_lr),
        "vae2": torch.optim.Adam(model["vae_model2"].parameters(), lr=args.vae_lr),
    }

    # %%
    train_loader = loader.run("warmup")

    # %%
    vae_model1 = model["vae_model1"]
    vae_model2 = model["vae_model2"]
    optimizer_vae1 = optimizers["vae1"]
    optimizer_vae2 = optimizers["vae2"]

    epoch = 0
    pbar = tqdm(desc="Epochs", total=args.num_epochs)
    while epoch < args.num_epochs + 1:
        lr = args.lr
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr
        eval_loader = loader.run("eval_train")

        if epoch < WARM_UP:
            warmup_trainloader = loader.run("warmup")
            logging.info("Warmup Net1")
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            logging.info("Warmup Net2")
            warmup(epoch, net2, optimizer2, warmup_trainloader)
        else:
            prob1, all_loss[0] = eval_train(net1, all_loss[0], eval_loader)
            prob2, all_loss[1] = eval_train(net2, all_loss[1], eval_loader)
            pred1 = prob1 > args.p_threshold
            pred2 = prob2 > args.p_threshold

            logging.info("Train Net1")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred2, prob2
            )  # co-divide
            loss_1 = train(
                epoch,
                net1,
                net2,
                optimizer1,
                labeled_trainloader,
                unlabeled_trainloader,
                train_loader,
                vae_model1,
                vae_model2,
                optimizer_vae1,
                device,
                net_1=True,
            )  # train net1

            logging.info("Train Net2")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred1, prob1
            )  # co-divide
            loss_2 = train(
                epoch,
                net2,
                net1,
                optimizer2,
                labeled_trainloader,
                unlabeled_trainloader,
                train_loader,
                vae_model2,
                vae_model1,
                optimizer_vae2,
                device,
                net_1=False,
            )  # train net2
        pbar.update()
        epoch += 1
    pbar.close()

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "net1_state_dict": net1.state_dict(),
            "net2_state_dict": net2.state_dict(),
            "vae1_state_dict": vae_model1.state_dict(),
            "vae2_state_dict": vae_model2.state_dict(),
            "optimizer1_state_dict": optimizer1.state_dict(),
            "optimizer2_state_dict": optimizer2.state_dict(),
            "loss_1": loss_1,
            "loss_2": loss_2,
        },
        checkpoint_file,
    )

    logging.info("Generating corrections")
    corrected_scores = []
    corrected_image_classes = []

    net1.eval()
    net2.eval()
    test_loader = loader.run("eval_train")
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.cuda()
            outputs1 = net1(images)
            outputs2 = net2(images)
            outputs = outputs1 + outputs2
            scores, predicted = torch.max(torch.softmax(outputs, 1), 1)
            corrected_scores += [score.item() for score in scores]
            corrected_image_classes += [loader.classes[idx] for idx in predicted]
    logging.info(f"Saving inference results for {len(corrected_scores)} images")
    corrections_file_path = checkpoint_file.parent / "corrections.json"
    image_paths = [
        Path(imgpath).name.split("-", maxsplit=1)[1] for imgpath in loader.image_paths
    ]
    with corrections_file_path.open("w") as corrections_file:
        json.dump(
            {
                "corrected_paths": image_paths,
                "original_labels": loader.image_classes,
                "corrected_labels": corrected_image_classes,
                "corrected_scores": corrected_scores,
            },
            corrections_file,
        )
