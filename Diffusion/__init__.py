from functools import partial
import torch
from ReSpace import SpacedDiffusion, space_timesteps
import GaussianDiffusion as gd


class IDDPM(SpacedDiffusion):

    def __init__(self, num_sampling_steps=None, timestep_respacing=None, noise_schedule="linear", use_kl=False, sigma_small=False, predict_xstart=False, learn_sigma=True, rescale_learned_sigmas=False, diffusion_steps=1000, cfg_scale=4.0, **kwargs):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE

        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]

        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart
                             else gd.ModelMeanType.START_X),
            model_var_type=((gd.ModelVarType.FIXED_LARGE if not sigma_small
                             else gd.ModelVarType.FIXED_SMALL)
                            if not learn_sigma else
                            gd.ModelVarType.LEARNED_RANGE),
            loss_type=loss_type,
            **kwargs)

        self.cfg_scale = cfg_scale

    def sample(
        self,
        model,
        text_encoder,
        z_size,
        prompts,
        device,
        additional_args=None,
        use_videoldm=False,
    ):
        n = len(prompts)
        z = torch.randn(n, *z_size, device=device)
        z = torch.cat([z, z], dim=0)

        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n).to(device=device)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        if use_videoldm:
            model_args['y'] = model_args['y'].squeeze()
            if 'mask' in model_args:
                # model_args['encoder_attention_mask'] = model_args['mask']
                del model_args['mask']

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale)

        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples


def forward_with_cfg(model, x, t, y, cfg_scale, cfg_channels=None, **kwargs):

    half = x[:len(x) // 2]
    combined = torch.cat([half, half], dim=0)

    for key in kwargs:
        if "mask" in key and kwargs[key] is not None:
            if len(kwargs[key]) != len(x):
                # repeat keys for cfg
                kwargs[key] = torch.cat([kwargs[key], kwargs[key]],
                                        dim=0).to(x.device)

    model_out = model.forward(combined, t, y, **kwargs)
    if not isinstance(model_out, torch.Tensor):
        model_out = model_out.sample
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    if cfg_channels is None:
        # cfg_channels = model_out.shape[1] // 2
        cfg_channels = 4  # use all channels for cfg


    eps, rest = model_out[:, :cfg_channels], model_out[:, cfg_channels:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)