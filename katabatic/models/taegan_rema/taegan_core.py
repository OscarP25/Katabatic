import json
import os
from typing import Tuple

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from .data_loader import TrainingDataLoader, InferenceDataLoader
from .utils import (
    AutoEncoder,
    Encoder,
    InteractionFactor,
    GradientPenalty,
    TableActivation,
    TableReconLoss,
)


class TAEGAN:
    def __init__(
        self,
        cache_dir: str,
        embed_dim: int = 256,
        cat_noise_dim: int = 64,
        cont_noise_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 6,
    ):
        with open(os.path.join(cache_dir, "span-info.json"), "r") as f:
            self.span_info = json.load(f)

        data_dim = sum(w for w, _ in self.span_info)

        self.generator = AutoEncoder(
            data_dim=data_dim,
            span_info=self.span_info,
            embed_dim=embed_dim,
            cat_noise_dim=cat_noise_dim,
            cont_noise_dim=cont_noise_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        self.discriminator = Encoder(
            in_dim=data_dim + len(self.span_info),
            out_dim=1,
            act=False,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        self.act = TableActivation(self.span_info)
        self.cache_dir = cache_dir

    # =========================================================
    # TRAIN
    # =========================================================
    def train(
        self,
        batch_size: int = 500,
        epochs: int = 300,
        warmup_epochs: int = 90,
        lr: float = 2e-4,
        l2scale: float = 1e-5,
        discrimination_steps: int = 2,
        generation_steps: int = 1,
        reconstruction_steps: int = 2,
        min_recon_weight: float = 0.1,
        max_recon_weight: float = 1.0,
        gp_lambda: float = 10,
    ):
        dataloader = TrainingDataLoader(self.cache_dir, batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator.to(device)
        self.discriminator.to(device)
        self.act.to(device)

        g_opt = Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(0.5, 0.9),
            eps=1e-3,
            weight_decay=l2scale,
        )

        d_opt = Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.9),
            eps=1e-3,
            weight_decay=l2scale,
        )

        g_scheduler = ExponentialLR(g_opt, gamma=0.99)
        d_scheduler = ExponentialLR(d_opt, gamma=0.99)

        gp = GradientPenalty(self.discriminator, gp_lambda)
        int_factor = InteractionFactor(self.span_info)

        loss_fct = TableReconLoss(
            span_info=self.span_info,
            min_weight=min_recon_weight,
            max_weight=max_recon_weight,
        )

        # ---------------- Warmup ----------------
        pbar = tqdm(desc="Warmup Train", total=warmup_epochs * len(dataloader))
        for _ in range(warmup_epochs):
            for d, m in dataloader:
                d, m = d.to(device), m.to(device)

                for _ in range(reconstruction_steps):
                    self._run_reconstruction_step(d, m, g_opt, loss_fct)

                for _ in range(discrimination_steps):
                    self._run_discrimination_step(d, m, d_opt, gp)

                pbar.update()

            d_scheduler.step()
            g_scheduler.step()

        # ---------------- Adversarial ----------------
        pbar = tqdm(desc="Train", total=epochs * len(dataloader))
        for _ in range(epochs):
            for d, m in dataloader:
                d, m = d.to(device), m.to(device)

                for _ in range(discrimination_steps):
                    self._run_discrimination_step(d, m, d_opt, gp)

                for _ in range(generation_steps):
                    self._run_generation_step(d, m, g_opt, int_factor, loss_fct)

                for _ in range(reconstruction_steps):
                    self._run_reconstruction_step(d, m, g_opt, loss_fct)

                pbar.update()

            d_scheduler.step()
            g_scheduler.step()

        torch.save(self.generator.state_dict(), os.path.join(self.cache_dir, "generator.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.cache_dir, "discriminator.pt"))

    # =========================================================
    # STEPS
    # =========================================================
    def _run_reconstruction_step(self, d, m, g_opt, loss_fct):
        g_opt.zero_grad()
        perm = torch.randperm(d.shape[0])

        fake = self.generator(d[perm], m[perm]).out
        recon_loss = loss_fct(fake, d[perm], m[perm])

        recon_loss.loss.backward(retain_graph=True)
        g_opt.step()

    def _run_discrimination_step(self, d, m, d_opt, gp):
        d_opt.zero_grad()

        perm_fake = torch.randperm(d.shape[0])
        perm_real = torch.randperm(d.shape[0])

        # 🔑 DETACH generator output
        fake = self.generator(d[perm_fake], m[perm_fake]).out.detach()
        fakeact = self.act(fake).detach()

        real_cat = torch.cat([d[perm_real], m[perm_fake].float()], dim=1)
        fake_cat = torch.cat([fakeact, m[perm_fake].float()], dim=1)

        d_real = -torch.mean(self.discriminator(real_cat).out)
        d_fake = torch.mean(self.discriminator(fake_cat).out)

        d_real.backward()
        d_fake.backward()
        gp(real_cat, fake_cat).backward()

        d_opt.step()

    def _run_generation_step(self, d, m, g_opt, int_factor, loss_fct):
        g_opt.zero_grad()

        perm_fake = torch.randperm(d.shape[0])
        perm_real = torch.randperm(d.shape[0])

        fake = self.generator(d[perm_fake], m[perm_fake]).out
        fakeact = self.act(fake)

        real_cat = torch.cat([d[perm_real], m[perm_real].float()], dim=1)
        fake_cat = torch.cat([fakeact, m[perm_fake].float()], dim=1)

        g_out = self.discriminator(fake_cat, with_info=True)
        g = -torch.mean(g_out.out)

        info_fake = g_out.info
        info_real = self.discriminator(real_cat, with_info=True).info

        loss_info_mean = torch.norm(info_fake.mean(0) - info_real.mean(0), 1)
        loss_info_std = torch.norm(info_fake.mean(0) - info_real.std(0), 1)

        real_int_mean, real_int_std = int_factor(d[perm_real])
        fake_int_mean, fake_int_std = int_factor(fakeact)

        loss_int_mean = torch.norm(real_int_mean - fake_int_mean, 1) / real_int_mean.shape[0]
        loss_int_std = torch.norm(real_int_std - fake_int_std, 1) / real_int_std.shape[0]

        recon_loss = loss_fct(fake, d[perm_fake], m[perm_fake])

        gen_loss = (
            g
            + recon_loss.loss
            + loss_info_mean
            + loss_info_std
            + loss_int_mean
            + loss_int_std
        )

        gen_loss.backward()
        g_opt.step()

        # =========================================================
    # GENERATE
    # =========================================================
    @torch.no_grad()
    def generate(
        self,
        n: int,
        batch_size: int = 100,
        temperature: Tuple[float, float] = (0.5, 1.2),
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataloader = InferenceDataLoader(n, self.cache_dir, batch_size)

        self.generator.load_state_dict(
            torch.load(
                os.path.join(self.cache_dir, "generator.pt"),
                map_location=device,
            )
        )
        self.generator.to(device).eval()
        self.act.eval()

        min_temp, max_temp = temperature
        all_data = []

        num_spans = len(self.span_info)

        pbar = tqdm(desc="Sampling", total=n)
        for d, m in dataloader:
            d = d.to(device)
            m = m.to(device)

            # m is (B, D) → convert to span-wise (B, S)
            # Each span has width w, so we collapse feature-mask → span-mask
            span_mask = []
            st = 0
            for w, _ in self.span_info:
                span_mask.append(m[:, st : st + w].any(dim=1))
                st += w
            span_mask = torch.stack(span_mask, dim=1).float()  # (B, S)

            # Iterate span-by-span
            for i in range(num_spans):
                sm = torch.zeros_like(span_mask)
                sm[:, i] = span_mask[:, i]

                fake = self.generator(d, sm, include_mask=True)
                mask = fake.mask.float()

                temp = (
                    (max_temp - min_temp)
                    * (1 - sm.mean(1, keepdim=True))
                    + min_temp
                )

                fakeact = self.act(fake.out, temperature=temp)
                d = fakeact * (1 - mask) + d * mask

            all_data.append(d)
            pbar.update(d.shape[0])

        return torch.cat(all_data, dim=0).cpu()
