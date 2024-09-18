import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
from torch import nn
import torch

class net(nn.Module):
    def __init__(self, input_dim, num_layers, hlsize,
                 conditional_shape=0):
        super(net, self).__init__()

        self.fc = [nn.Linear(input_dim+1+conditional_shape, hlsize),
                     nn.ReLU()]
        for i in range(num_layers):
            self.fc.append(nn.Linear(hlsize, hlsize))
            self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(hlsize, input_dim))

        self.network = nn.Sequential(*self.fc)
        
    def forward(self, x):
        x = self.network(x)
        return x
    
class FlowMatch():
    def __init__(self, data, input_dim, num_layers, hlsize,
                 lr=0.001, batch_size=1000, conditional=None):
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hlsize = hlsize
        self.lr = lr
        self.batch_size = batch_size

        self.data = data
        self.conditional = conditional
        if self.conditional is not None:
            self.conditional_shape = self.conditional.shape[1]
        self.noise_distribution = np.random.normal(0, 1, self.data.shape)

        self.model = net(self.input_dim, self.num_layers, self.hlsize,
                         conditional_shape=self.conditional_shape)
    
    def train(self, epochs=500, patience=100, sigma0=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        def loss(x0, x1, sigma0, conditional=None):

            # x0 is noise and x1 is the data distribution
            x0 = torch.tensor(x0)
            x1 = torch.tensor(x1)
            t = torch.rand((self.batch_size, 1))
            noise = torch.randn((self.batch_size, 2))
            # the mean of the probability path P(x|z)
            # interpolation between noise and data distribution
            # where x0 is the noise distribution and x1 is the data distribution
            psi0 = t * x1 + (1 - t) * x0 
            # the standard deviation is just a constant times standard normal
            smoothing = sigma0 * noise
            # adding the gaussian noise to the mean
            psi0 = psi0 + smoothing
            if conditional is None:
                output = self.model(torch.tensor(
                    np.column_stack((psi0, t)).astype(np.float32)))
            else:
                output = self.model(torch.tensor(
                    np.column_stack((psi0, 
                        t, conditional)).astype(np.float32)))
            psi = x1 - x0
            return (output - psi).pow(2).mean()

        if self.conditional is None:
            train_dataset = TensorDataset(torch.tensor(
                np.column_stack(
                    (self.noise_distribution, self.data)).astype(np.float32)))
        else:
            train_dataset = TensorDataset(torch.tensor(
                np.column_stack(
                    (self.noise_distribution, self.data, 
                     self.conditional)).astype(np.float32)))
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, shuffle=True)

        c = np.inf
        count = 0
        self.loss_history = []
        for i in range(epochs):
            loss_over_batch = []
            for batch_X in train_loader:
                optimizer.zero_grad()
                if self.conditional is None:
                    l = loss(batch_X[0][:, :self.data.shape[1]], 
                             batch_X[0][:, self.data.shape[1]:], sigma0)
                else:
                    l = loss(batch_X[0][:, :self.data.shape[1]],
                            batch_X[0][:, self.data.shape[1]:2*self.data.shape[1]],
                            sigma0, batch_X[0][:, 2*self.data.shape[1]:])
                l.backward()
                optimizer.step()
                loss_over_batch.append(l.item())
            print('Epoch: ', str(i) + ' Loss: ', 
                np.mean(loss_over_batch), ' Best Loss: ', 
                c, ' Count: ', count)
            self.loss_history.append(np.mean(loss_over_batch))
            if self.loss_history[-1] < c:
                c = self.loss_history[-1]
                best_model = self.model.state_dict()
                count = 0
            else:
                count += 1
            if count > patience:
                break

        self.model.load_state_dict(best_model)

    def sample(self, nsamples, time_samples=10):
        def func(t, x):
            # hmm not sure about this... might not need the psi0 bit...
            x = x.reshape(-1, 2)
            t = np.array([t]*len(x)).reshape(-1, 1)
            return self.model(torch.tensor(np.column_stack((x, 
                    t)).astype(np.float32))).detach().numpy().flatten()
        
        test_noise_distribution = np.random.normal(0, 1, 
                                    (nsamples, self.data.shape[1]))

        t = np.linspace(0, 1, time_samples)
        sol = solve_ivp(func, (0, 1), test_noise_distribution.flatten(), t_eval=t)
        return sol.y.T.reshape(time_samples, nsamples, self.data.shape[1])
    
    def log_p_base(self, x, reduction='sum', dim=1):
        log_p = -0.5 * torch.log(2. * torch.tensor(np.pi)) - 0.5 * x**2.
        if reduction == 'mean':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p
    
    def log_prob(self, x_1, reduction=None, time_samples=10):
        # backward Euler (see Appendix C in Lipman's paper)
        ts = torch.linspace(1., 0., time_samples)
        delta_t = ts[1] - ts[0]

        if type(x_1) is not torch.Tensor:
            x_1 = torch.tensor(x_1, dtype=torch.float32)
        
        for t in ts:
            if t == 1.:
                x_t = x_1 * 1.
                f_t = 0.
            else:
                # Calculate phi_t
                t_embedding = torch.Tensor([t]*len(x_t)).reshape(-1, 1)
                data_stack = torch.cat((x_t, t_embedding), dim=1)
                x_t =x_t - self.model(data_stack) * delta_t
                
                # Calculate f_t
                # approximate the divergence using the Hutchinson trace estimator and the autograd
                self.model.eval()  # set the vector field net to evaluation
                
                x = torch.FloatTensor(data_stack.data)  # copy the original data (it doesn't require grads!)
                x.requires_grad = True 
                
                e = torch.randn_like(x)  # epsilon ~ Normal(0, I) 
                
                e_grad = torch.autograd.grad(self.model(x).sum(), x, create_graph=True)[0]
                e_grad_e = e_grad * e
                f_t = e_grad_e.view(x.shape[0], -1).sum(dim=1)

                self.model.eval()  # set the vector field net to train again
        
        log_p_1 = self.log_p_base(x_t, reduction='sum') - f_t
        
        if reduction == "mean":
            return log_p_1.mean()
        elif reduction == "sum":
            return log_p_1.sum()
        else:
            return log_p_1
    
