# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Create the spiral
def swiss_roll_data(size, noise=0.05):
    x, color = make_swiss_roll(size, noise=noise)
    data = (x[:, [0, 2]] / 10.0).T
    return data, color

# Create the two clusters
def two_clusters_data(size, spacing=0.2):
    cluster = np.random.randint(2, size=size) * spacing - (spacing / 2)
    points = np.random.random(size) - 0.5
    data = np.array([[cluster, points] for cluster, points in zip(cluster, points)]).T
    return data, cluster

def denoising_score_matching(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss
        
# Forward Process Training process
def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def get_coef_at_step_t(input_, t, x, device):
    t = t.to(device)
    input_ = input_.to(device)
    shape = x.shape
    out = torch.gather(input_, 0, t)

    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


# Fixed computation of variables for all timesteps
def get_variables(betas, device):
    alphas = (1 - betas).to(device)
    alphas_prod = torch.cumprod(alphas, 0).to(device)
    alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)
    
    return alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt

# Sampling function
def q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alpha_t = get_coef_at_step_t(alphas_bar_sqrt, t, x_0, device)
    alpha_1_m_t = get_coef_at_step_t(one_minus_alphas_bar_sqrt, t, x_0, device)
    return (alpha_t * x_0 + alpha_1_m_t * noise)

def plot_diffusion(dataset, n_steps, color, device):
    betas = make_beta_schedule(schedule='sigmoid', n_timesteps=n_steps, start=1e-5, end=1e-2)
    alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt = get_variables(betas, device)
    
    n_plots = 20
    step_size = n_steps//n_plots
    fig, axs = plt.subplots(n_plots//10, 10, figsize=(18, 3.7))
    plt.suptitle("Forward Diffusion", y=1.03, fontsize=20)
    for i in range(n_plots):
        cur_step = i * step_size
        q_i = q_sample(dataset, torch.tensor([cur_step]), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device)
        q_i = q_i.cpu().detach().numpy()
        axs[i // 10, i % 10].scatter(q_i[:, 0], q_i[:, 1], s=1.5, c=color) 
        axs[i // 10, i % 10].set_axis_off()
        axs[i // 10, i % 10].set_title('$q(\mathbf{x}_{'+str(cur_step)+'})$')

def p_mean_variance(model, x, t, device):
    # Go through model
    out = model(x, t).to(device)
    # Extract the mean and variance
    mean, log_var = torch.split(out, 2, dim=-1)
    var = torch.exp(log_var).to(device)
    return mean, log_var 

def p_sample(model, x, t, alphas, one_minus_alphas_bar_sqrt, betas, device):
    t = torch.tensor([t]).to(device)
    # Factor to the model output
    eps_factor = ((1 - get_coef_at_step_t(alphas, t, x, device)) / get_coef_at_step_t(one_minus_alphas_bar_sqrt, t, x, device))
    # Model output
    eps_theta = model(x, t)
    # Final values
    mean = (1 / get_coef_at_step_t(alphas, t, x, device).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x).to(device)
    # Fixed sigma
    sigma_t = get_coef_at_step_t(betas, t, x, device).sqrt()
    sample = mean + sigma_t * z
    return (sample)

def p_sample_loop(dataset, model, n_steps, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, betas, device, clusters_cur_x=None):
    shape = dataset.shape
    if clusters_cur_x is not None:
        cur_x = clusters_cur_x
    else:
        cur_x = q_sample(dataset, torch.tensor([n_steps-1]), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device).to(device) # altered this line to keep track of the color. Used to be pure noise torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, alphas, one_minus_alphas_bar_sqrt, betas, device)
        x_seq.append(cur_x)
    return x_seq

def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return kl

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(torch.tensor(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, device, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1 - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(torch.clamp(cdf_delta, min=1e-12)))).to(device)
    return log_probs


def noise_estimation_loss(model, x_0, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device):
    batch_size = x_0.shape[0]
    
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,)).to(device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long().to(device)
    # x0 multiplier
    a = get_coef_at_step_t(alphas_bar_sqrt, t, x_0, device)
    # eps multiplier
    am1 = get_coef_at_step_t(one_minus_alphas_bar_sqrt, t, x_0, device)
    e = torch.randn_like(x_0).to(device)
    # model input
    x = (a * x_0 + am1 * e)
    output = model(x, t).to(device)
        
    return (e - output).square().mean()





class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
    
class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = nn.Linear(128, 2)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)
    


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
    
    
    
def get_color_point_for_clusters(clusters_diffused, n_steps, color, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device):
    clusters_diffused[-1] = [0, 0]
    COLOR_POINT = color
    COLOR_POINT[-1] = 0.22
    SIZE_POINT = np.ones(len(clusters_diffused)) * 2
    SIZE_POINT[-1] = 50
    
    return clusters_diffused, COLOR_POINT, SIZE_POINT
        
        
def train(dataset, n_steps, device, batch_size=128, epochs=101, color='red', size=5, clusters=False):
    model = ConditionalModel(n_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_data_points = dataset.size()[0]
    
    betas = make_beta_schedule(schedule='sigmoid', n_timesteps=n_steps, start=1e-5, end=1e-2)
    alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt = get_variables(betas, device)
    
    # Create EMA model
    ema = EMA(0.9)
    ema.register(model)

    for t in range(epochs):
        # X is a torch Variable
        permutation = torch.randperm(n_data_points).to(device)
        for i in range(0, n_data_points, batch_size):
            # Retrieve current batch
            indices = permutation[i:i+batch_size]
            batch_x = dataset[indices].to(device)
            # Compute the loss.
            loss = noise_estimation_loss(model, batch_x, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device).to(device)
            # Before the backward pass, zero all of the network gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to parameters
            loss.backward()
            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # Calling the step function to update the parameters
            optimizer.step()
            # Update the exponential moving average
            ema.update(model)

        # Print loss
        if (t % (epochs // 10) == 0):
            print(f'Epoch {t}: loss = {round(loss.item(), 4)}')
            x_seq = p_sample_loop(dataset, model, n_steps, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, betas, device)
            
            if clusters:
                clusters_diffused = x_seq[0].detach().cpu().numpy()
                clusters_cur_x, color, size = get_color_point_for_clusters(clusters_diffused, n_steps, color, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device)
                x_seq = p_sample_loop(dataset, model, n_steps, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, betas, device, clusters_cur_x=torch.tensor(clusters_cur_x).to(device))
            
            PLOTS = 10
            fig, axs = plt.subplots(1, PLOTS, figsize=(32, 3))
            for i in range(PLOTS):
                if i == PLOTS-1:
                    idx = n_steps-1
                else:  
                    idx = i * (n_steps // PLOTS)
                    
                cur_x = x_seq[idx].detach().cpu().numpy()       

                axs[i].scatter(cur_x[:, 0], cur_x[:, 1], c=color, s=size)
                axs[i].set_title(f'epoch {t} time step {n_steps-1 - (idx)}')

    return model