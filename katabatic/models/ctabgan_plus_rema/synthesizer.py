import numpy as np
import torch
from torch.nn import (
    Module, Sequential, Linear, Conv2d, ConvTranspose2d,
    LeakyReLU, ReLU, Dropout, Sigmoid
)
from torch.nn import functional as F
from torch.optim import Adam

# Katabatic imports
from katabatic.models.ctabgan_plus.transformer import (
    DataTransformer,
    ImageTransformer
)



class Classifier(Module):
    def __init__(self, input_dim, dis_dims, st_ed):
        super().__init__()
        dim = input_dim - (st_ed[1] - st_ed[0])
        seq = []
        self.str_end = st_ed

        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        if (st_ed[1] - st_ed[0]) == 1:
            seq += [Linear(dim, 1)]
        elif (st_ed[1] - st_ed[0]) == 2:
            seq += [Linear(dim, 1), Sigmoid()]
        else:
            seq += [Linear(dim, (st_ed[1] - st_ed[0]))]

        self.seq = Sequential(*seq)

    def forward(self, input):
        if (self.str_end[1] - self.str_end[0]) == 1:
            label = input[:, self.str_end[0]:self.str_end[1]]
        else:
            label = torch.argmax(
                input[:, self.str_end[0]:self.str_end[1]], axis=-1
            )

        new_imp = torch.cat(
            (input[:, :self.str_end[0]], input[:, self.str_end[1]:]), 1
        )

        if (self.str_end[1] - self.str_end[0]) in [1, 2]:
            return self.seq(new_imp).view(-1), label
        return self.seq(new_imp), label


def apply_activate(data, output_info):
    data_t, st = [], 0
    for item in output_info:
        ed = st + item[0]
        if item[1] == 'tanh':
            data_t.append(torch.tanh(data[:, st:ed]))
        elif item[1] == 'softmax':
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
        st = ed
    return torch.cat(data_t, dim=1)


def get_st_ed(target_col_index, output_info):
    st, c, tc = 0, 0, 0
    for item in output_info:
        if c == target_col_index:
            break
        st += item[0]
        if item[1] == 'softmax' or (item[1] == 'tanh' and item[2] == 'yes_g'):
            c += 1
        tc += 1
    return (st, st + output_info[tc][0])


class Discriminator(Module):
    def __init__(self, layers):
        super().__init__()
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, layers):
        super().__init__()
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)



class CTABGANSynthesizer:

    def __init__(
        self,
        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        l2scale=1e-5,
        batch_size=500,
        epochs=300
    ):
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    # TRAIN
   
    def fit(self, train_data, categorical, mixed, general, non_categorical, type):

        problem_type, target_index = None, None
        if type:
            problem_type = list(type.keys())[0]
            target_index = train_data.columns.get_loc(type[problem_type])

        # Data transform 
        self.transformer = DataTransformer(
            train_data=train_data,
            categorical_list=categorical,
            mixed_dict=mixed,
            general_list=general,
            non_categorical_list=non_categorical,
        )
        self.transformer.fit()

        data = self.transformer.transform(train_data.values)
        data = torch.tensor(data, dtype=torch.float32, device=self.device)

        data_dim = self.transformer.output_dim
        side = self._find_side(data_dim)

        self.Gtransformer = ImageTransformer(side)
        self.Dtransformer = ImageTransformer(side)

        # Models
        self.generator = Generator(self._gen_layers()).to(self.device)
        self.discriminator = Discriminator(self._dis_layers()).to(self.device)

        self.classifier = None
        if target_index is not None:
            self.classifier = Classifier(
                data_dim,
                self.class_dim,
                get_st_ed(target_index, self.transformer.output_info)
            ).to(self.device)

        #Optimizers
        self.opt_g = Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.9),
            weight_decay=self.l2scale
        )

        d_params = list(self.discriminator.parameters())
        if self.classifier:
            d_params += list(self.classifier.parameters())

        self.opt_d = Adam(
            d_params,
            lr=2e-4,
            betas=(0.5, 0.9),
            weight_decay=self.l2scale
        )

        # Training loop 
        for epoch in range(self.epochs):

            perm = torch.randperm(len(data))

            for i in range(0, len(data), self.batch_size):

                real = data[perm[i:i + self.batch_size]]

                
                noise = torch.randn(
                    len(real), self.random_dim, 1, 1, device=self.device
                )

                fake = self.generator(noise)
                fake = self.Gtransformer.inverse_transform(fake)
                fake = apply_activate(fake, self.transformer.output_info)

                real_img = self.Dtransformer.transform(real)
                fake_img = self.Dtransformer.transform(fake.detach())

                d_real = self.discriminator(real_img)
                d_fake = self.discriminator(fake_img)

                loss_d = (
                    torch.mean(F.relu(1. - d_real)) +
                    torch.mean(F.relu(1. + d_fake))
                )

                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()

                
                fake_img = self.Dtransformer.transform(fake)
                g_fake = self.discriminator(fake_img)

                loss_g = -torch.mean(g_fake)

                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()

    
    def sample(self, n):

        self.generator.eval()
        steps = n // self.batch_size + 1
        data = []

        for _ in range(steps):
            noise = torch.randn(
                self.batch_size, self.random_dim, 1, 1, device=self.device
            )
            fake = self.generator(noise)
            fake = self.Gtransformer.inverse_transform(fake)
            fake = apply_activate(fake, self.transformer.output_info)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result, _ = self.transformer.inverse_transform(data)
        return result[:n]

    # Helpers

    def _find_side(self, dim):
        for s in [4, 8, 16, 32, 64]:
            if s * s >= dim:
                return s
        return 64

    # Generator now outputs side × side
    def _gen_layers(self):
        layers = [
            ConvTranspose2d(self.random_dim, 256, 4, 1, 0),
            ReLU(True),
        ]

        curr_size = 4
        curr_channels = 256

        while curr_size < self.Gtransformer.height:
            next_channels = max(curr_channels // 2, 64)
            layers += [
                ConvTranspose2d(curr_channels, next_channels, 4, 2, 1),
                ReLU(True)
            ]
            curr_channels = next_channels
            curr_size *= 2

        layers += [Conv2d(curr_channels, 1, 3, 1, 1)]
        return layers

    #  Discriminator FC dimension correct for any side
    def _dis_layers(self):
        h = self.Dtransformer.height // 4
        return [
            Conv2d(1, 64, 4, 2, 1),
            LeakyReLU(0.2),
            Conv2d(64, 128, 4, 2, 1),
            LeakyReLU(0.2),
            torch.nn.Flatten(),
            Linear(128 * h * h, 1)
        ]
