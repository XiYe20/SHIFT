# =========================
# WAN2.2 Flow-Matching utils
# =========================
import math
import torch


class FlowMatchSchedulerWan:
    """
    Copied from DiffSynth-Studio (FlowMatchScheduler, template='Wan'), simplified to only keep WAN.
    Discrete schedule with 1000 train timesteps by default.
    """
    def __init__(self):
        self.num_train_timesteps = 1000
        self.training = False
        self.sigmas = None
        self.timesteps = None
        self.linear_timesteps_weights = None

    @staticmethod
    def set_timesteps_wan(num_inference_steps=1000, denoising_strength=1.0, shift=5):
        """
        DiffSynth-Studio only perform finetuning on the discrete sigmas we will use in the inference process
        Align the train-test behaviour. Thus, the training timesteps depends on num_inference_steps. 
        """
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000

        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        # Copied from DiffSynth; produces a per-timestep weight curve
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        weights = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def set_timesteps(self, num_inference_steps=1000, denoising_strength=1.0, training=False, shift=5):
        self.sigmas, self.timesteps = self.set_timesteps_wan(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=shift,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def _timestep_id(self, timestep):
        # timestep is a scalar tensor (float), match to closest discrete timestep
        if isinstance(timestep, torch.Tensor):
            t_cpu = timestep.detach().to("cpu")
        else:
            t_cpu = torch.tensor(float(timestep))
        return torch.argmin((self.timesteps - t_cpu).abs())

    def add_noise(self, original_samples, noise, timestep):
        # x_t = (1 - sigma) * x + sigma * eps
        tid = self._timestep_id(timestep)
        sigma = self.sigmas[tid].to(device=original_samples.device, dtype=original_samples.dtype)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        # Flow matching target
        return noise - sample

    def training_weight(self, timestep):
        if self.linear_timesteps_weights is None:
            raise RuntimeError("training_weight called before set_timesteps(..., training=True)")
        tid = self._timestep_id(timestep.to(self.timesteps.device))
        return self.linear_timesteps_weights[tid].to(device=timestep.device, dtype=timestep.dtype)


def flow_match_sft_loss_wan(
    scheduler: FlowMatchSchedulerWan,
    model_pred: torch.Tensor,
    input_latents: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
):
    """
    Equivalent to DiffSynth FlowMatchSFTLoss core:
    mse(pred, noise - input_latents) * weight(t)
    """
    target = scheduler.training_target(input_latents, noise, timestep)
    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
    loss = loss * scheduler.training_weight(timestep)
    return loss