import torch
import matplotlib.pyplot as plt
from first_principles_models import LinearSigmoidModel
from first_principles_models import LinearModel
from data import get_logo_as_rgb_image_tensor
from utilities import plot_scalar_image_tensor

RED, GREEN, BLUE = (0, 1, 2)
"""
loss functie heeft invloed op de gradient,hoe snel en wat je leert
"""
# LinearModel
linear_learning_rate = 0.0001
image = get_logo_as_rgb_image_tensor()
target = torch.tensor([0.25, 0.75], dtype=torch.float32)
torch.manual_seed(0)
linear_model = LinearModel(image.shape, target.shape)

linear_initial_output = linear_model.calculate_output(image)
linear_initial_loss = linear_model.calculate_loss(linear_initial_output, target)
linear_gradients = linear_model.calculate_gradients(image, linear_initial_output, target)

linear_model.update_weights(linear_gradients, linear_learning_rate)
linear_output_1 = linear_model.calculate_output(image)
linear_gradients_1 = linear_model.calculate_gradients(image, linear_output_1, target)
linear_model.update_weights(linear_gradients_1, linear_learning_rate)

plots = [
    (linear_gradients_1[..., RED, 0], "Gradient: RED, output 0"),
    (linear_gradients_1[..., GREEN, 0], "Gradient: GREEN, output 0"),
    (linear_gradients_1[..., BLUE, 0], "Gradient: BLUE, output 0"),
    (linear_gradients_1[..., RED, 1], "Gradient: RED, output 1"),
    (linear_gradients_1[..., GREEN, 1], "Gradient: GREEN, output 1"),
    (linear_gradients_1[..., BLUE, 1], "Gradient: BLUE, output 1"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (tensor, title) in zip(axes.flat, plots):
    im = ax.imshow(tensor.detach().cpu().numpy(), cmap="RdBu_r")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Linear Model – Gradients", fontsize=16)
plt.tight_layout()
plt.savefig("Linear_Gradients.png", dpi=120)
plt.show()

##################

# LinearSigmoidModel
sigmoid_learning_rate = 0.2
image = get_logo_as_rgb_image_tensor()
target = torch.tensor([0.25, 0.75], dtype=torch.float32)

torch.manual_seed(0)
sigmoid_model = LinearSigmoidModel(image.shape, target.shape)

sigmoid_initial_output = sigmoid_model.calculate_output(image)
sigmoid_initial_loss = sigmoid_model.calculate_loss(sigmoid_initial_output, target)
sigmoid_gradients = sigmoid_model.calculate_gradients(image, sigmoid_initial_output, target)

sigmoid_model.update_weights(sigmoid_gradients, sigmoid_learning_rate)
sigmoid_output_1 = sigmoid_model.calculate_output(image)
sigmoid_gradients_1 = sigmoid_model.calculate_gradients(image, sigmoid_output_1, target)
sigmoid_updated_weights = sigmoid_model.update_weights(sigmoid_gradients_1, sigmoid_learning_rate)

plots = [
    (sigmoid_gradients_1[..., RED, 0], "Gradient: RED, output 0"),
    (sigmoid_gradients_1[..., GREEN, 0], "Gradient: GREEN, output 0"),
    (sigmoid_gradients_1[..., BLUE, 0], "Gradient: BLUE, output 0"),
    (sigmoid_gradients_1[..., RED, 1], "Gradient: RED, output 1"),
    (sigmoid_gradients_1[..., GREEN, 1], "Gradient: GREEN, output 1"),
    (sigmoid_gradients_1[..., BLUE, 1], "Gradient: BLUE, output 1"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (tensor, title) in zip(axes.flat, plots):
    im = ax.imshow(tensor.detach().cpu().numpy(), cmap="RdBu_r")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Linear Sigmoid Model – Gradients", fontsize=16)
plt.tight_layout()
plt.savefig("Sigmoid_Gradients.png", dpi=120)
plt.show()

##################

# Learning rates
image = get_logo_as_rgb_image_tensor()
target = torch.tensor([0.25, 0.75], dtype=torch.float32)

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
num_steps = 10

loss_curves = {}

for lr in learning_rates:
    torch.manual_seed(0)
    model = LinearSigmoidModel(image.shape, target.shape)

    losses = []
    for step in range(num_steps):
        output = model.calculate_output(image)
        loss = model.calculate_loss(output, target)
        losses.append(float(loss))  # loss BEFORE this update

        gradients = model.calculate_gradients(image, output, target)
        model.update_weights(gradients, lr)

    # loss after the final update, so we show the effect of all 10 steps
    final_output = model.calculate_output(image)
    losses.append(float(model.calculate_loss(final_output, target)))

    loss_curves[lr] = losses

plt.figure(figsize=(8, 5))
for lr, losses in loss_curves.items():
    plt.plot(range(len(losses)), losses, marker="o", label=f"rate = {lr}")

plt.xlabel("Update step")
plt.ylabel("Training loss")
plt.title("LinearSigmoidModel – Training loss for different learning rates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Sigmoid_Learning_Rates.png", dpi=120)
plt.show()
