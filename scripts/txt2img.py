import argparse, os, sys, glob
import torch
import numpy as np
import PIL
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from einops import rearrange, repeat

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from pytorch_lightning import seed_everything

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs_final/accordion/"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")
    
    parser.add_argument("--content_path", type=str, help="Path to content image")
    parser.add_argument("--strength", default=0.99, type=float, help="content percent")
    parser.add_argument("--seed", default=23, type=int, help="random seed")
    opt = parser.parse_args()
   
    seed_everything(opt.seed)
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    model.embedding_manager.load(opt.embedding_path)
    batch_size = opt.n_samples

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    for file in os.listdir(opt.content_path):

        content_name = opt.content_path.split('/')[-1].split('.')[0]
        content_image = load_img(os.path.join(opt.content_path, file)).to(device)
        content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
        content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

        init_latent = content_latent

        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        prompt = opt.prompt


        sample_path = os.path.join(outpath, content_name)
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = batch_size * [""]
                    # uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    # c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    c = prompt * batch_size

                    t_enc = torch.tensor([int(opt.strength * 1000)], dtype=torch.int64).to(device)
                    c_enc = model.get_learned_conditioning(batch_size * prompt, t_enc)
                    x_noisy = model.q_sample(x_start=init_latent, t=t_enc)
                    model_output = model.apply_model(x_noisy, t_enc, c_enc)
                    z_enc = sampler.stochastic_encode(init_latent, t_enc,\
                                                            noise = model_output, use_original_steps = True)
                    t_enc = torch.tensor([int(opt.strength * opt.ddim_steps)], dtype=torch.int64).to(device)
                    samples = sampler.decode(z_enc, c, t_enc, 
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,
                                            get_learned_conditioning=model.get_learned_conditioning)

                    x_samples_ddim = model.decode_first_stage(samples)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        save_path = os.path.join(sample_path, file)
                        Image.fromarray(x_sample.astype(np.uint8)).save(save_path)
      
                        base_count += 1
                    all_samples.append(x_samples_ddim)


        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        save_path = os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg')

        
        print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
