import torch
from transformers import Trainer, TrainerCallback
from transformers.integrations import WandbCallback
import matplotlib.pyplot as plt

class TrainerCustomCallback(TrainerCallback):
    def on_log(self, args, state, control, model, **kwargs):
        kwargs["logs"]["weight_norm_cross_attention"] = sum(p.data.norm(2).item() ** 2 for name, p in model.named_parameters() if "crossattention" in name) ** 0.5
        kwargs["logs"]["weight_norm_decoder"] = sum(p.data.norm(2).item() ** 2 for name, p in model.decoder.named_parameters() if "crossattention" not in name) ** 0.5
        if model.encoder is not None:
            kwargs["logs"]["weight_norm_encoder"] = sum(p.data.norm(2).item() ** 2 for name, p in model.encoder.named_parameters() if "crossattention" not in name) ** 0.5
        # log hamming and levenshtein distances from the input of the forward pass at each training step
        kwargs["logs"]["hamming_dist"] = torch.median(model.hamming_dist).item()
        kwargs["logs"]["levenshtein_dist"] = torch.median(model.levenshtein_dist).item()
        kwargs["logs"]["masking_fraction"] = model.masking_fraction
        if model.rescale_loss_diffusion and ("loss" in kwargs["logs"]):
            kwargs["logs"]["loss_rescaled"] = kwargs["logs"]["loss"]
            kwargs["logs"]["loss"] = kwargs["logs"]["loss"]*kwargs["logs"]["masking_fraction"]
        if model.rescale_loss_diffusion and ("eval_loss" in kwargs["logs"]):
            kwargs["logs"]["eval_loss_rescaled"] = kwargs["logs"]["eval_loss"]
            kwargs["logs"]["eval_loss"] = kwargs["logs"]["eval_loss"]*kwargs["logs"]["masking_fraction"]
        
        
class WandbCustomCallback(WandbCallback):
    """Custom WandbCallback to log parameters of the model and hamming and levenshtein
    distances between input and context sequences.
    """
    def __init__(self, trainer, show_cross_attentions=False):
        super().__init__()
        self.trainer = trainer
        self.show_cross_attentions = show_cross_attentions

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # get the parameters of the model
        par, par1, par2 = {}, {}, {}
        for i in range(len(self.trainer.model.decoder.esm.encoder.layer)):
            if hasattr(self.trainer.model.decoder.esm.encoder.layer[i], "crossattention"):
                par[i] = self.trainer.model.decoder.esm.encoder.layer[i].crossattention.weight_cross_attention.item()
                par1[i] = self.trainer.model.decoder.esm.encoder.layer[i].weight_crossattention_ffw.item()
                par2[i] = sum(p.data.norm(2).item() ** 2 for name, p in self.trainer.model.decoder.esm.encoder.layer[i].named_parameters() if "crossattention" in name) ** 0.5
        # log the parameters on wandb separately
        for i in par.keys():
            self._wandb.log({f"crossattention_weight_{i}": par[i]})
            self._wandb.log({f"ffw_weight_{i}": par1[i]})
            self._wandb.log({f"crossattention_norm_{i}": par2[i]})
        
        if self.show_cross_attentions:
            # get the cross-attention matrices
            batch = next(iter(eval_dataloader))
            # get only first sample of batch
            for k in batch.keys():
                if batch[k] is not None:
                    batch[k] = batch[k][:1]
            with torch.no_grad():
                outputs = model(**batch, output_attentions=True, return_dict=False)
                cross_attentions = outputs[-1]
            # log the attention matrices
            for i in range(len(cross_attentions)):
                # cross_attentions[i][0] is a tensor with the 20 attention heads of the i-th layer
                # the shape is (20,N,M) where N is the length of the input sequence and M is the length of the context sequence
                # log it as a plot using pyplot to make it easier to visualize
                fig, axs = plt.subplots(cross_attentions[i][0].shape[0]//5, 5,
                                        figsize=(5*5, cross_attentions[i][0].shape[0]//5*5),
                                        constrained_layout=True, sharex=True, sharey=True)
                for j in range(cross_attentions[i][0].shape[0]):
                    ax = axs[j//5, j%5]
                    ax.imshow(cross_attentions[i][0][j].cpu().numpy())
                self._wandb.log({f"cross_attention_{i}": fig})
                plt.close(fig)

from torch.profiler import profile, ProfilerActivity
import wandb

class ProfilingTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir"),
            # record_shapes=True,
            # profile_memory=True
        ) as prof:
            result = super().training_step(*args, **kwargs)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        return result