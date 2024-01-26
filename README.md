# AIGC Generative Models

For NLP generative, like GPT, please check https://github.com/rmgogogo/nano-transformers

Here this repo more on generatives. GPT still may be tried here.

This repo uses PyTorch.

## VAE

```
python vae.py --train --epochs 10 --predict
```

![](doc/vae.png)

## Conditional VAE

```
python cvae.py --train --epochs 10 --predict
```

![](doc/cvae.png)

## Diffusion

```
python diffusion.py --train --epochs 100 --predict
```

Mac Mini M1 takes around 1 hour (1:17:16).

![](doc/diffusion.png)


## Conditional Diffusion

```
python conditional_diffusion.py --train --epochs 100 --predict
```

![](doc/conditional_diffusion.png)

## CLIP

```
python clip.py --train --epochs 10 --predict
```

![](doc/clip.png)

## CLIP Pro

A pro version of CLIP. It uses the BERT text encoder with real text.
Since this is a nano image VAE, while BERT encoder generates 768-d vector, and we only have 10 ditigals, it has high prob to contain same digital in one batch, then the CLIP's loss can't work well. Using small batch would help but small batch has its own problem. So the performance is not good.
However it's good enough as a demo to tell the essience.

```
python clip_pro.py --train --epochs 10 --predict
```

![](doc/clip_pro.png)

## VQ VAE

```
python vqvae.py --train --epochs 100 --predict
```

Codebook size is 32, here display the whole possibilites. This sample VQ the whole z, in real case, it VQ the parts.

![](doc/vqvae.png)

The initial codebook:

![](doc/vqvae-init-cb.png)

The learned codebook:

![](doc/vqvae-learned-cb.png)

## DDIM (Faster Diffusion Generation)

50 times faster.

```
python diffusion.py --predict --ddim
python conditional_diffusion.py --predict --ddim
```

![](doc/diffusion_ddim.png)

![](doc/conditional_diffusion_ddim.png)

## Latent Diffusion

Based on vae with latent 8, it do diffusion in latent space.
However since the latent space already is noise-make-sense and high compressed (8 numbers), the diffusion in latent didn't work well as expected.
It's mainly for demo purpose.

![](doc/latent_diffusion.png)