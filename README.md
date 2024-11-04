# Theseus: A Basic T2V Model Inspired by CogVideoX

This repository contains a very basic implementation of the ideas presented in the CogVideoX paper, including 3D attention mechanisms and causal convolutional layers. The results shown were obtained after training for 100 epochs on a mini-dataset of 10 videos using a NVIDIA T4 GPU on Google Colab.

## Overview

Theseus is a simplified Text-to-Video (T2V) model that leverages core concepts from the CogVideoX architecture. While not a complete implementation, it provides a foundational framework to explore the intersection of text and video modalities through deep learning.

# 🛠 Theseus 2.0: DiT for T2V models

Complete update of the model, and the results of the new implementation will be presented soon :b

## Well now I have this error when I start training xd

```
[2024-11-04 22:46:16] Experiment directory created at outputs/028
Total samples accepted: 10
[2024-11-04 22:46:16] Dataset contains 10 videos (/teamspace/studios/this_studio/data/dataset.csv)
[2024-11-04 22:46:16] Batch size: 2
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Trainable params: 149121056, Total params: 350656544
[2024-11-04 22:47:04] Beginning epoch 0...
[2024-11-04 22:47:19] Reference video stats - Shape: torch.Size([3, 16, 256, 1088]), Min: -1.0000, Max: 1.0000, Mean: -0.6577
[2024-11-04 22:47:19] 
=== Video Tensor Analysis ===
[2024-11-04 22:47:19] Shape: torch.Size([3, 16, 256, 1088])
[2024-11-04 22:47:19] Range: [-1.0000, 1.0000]
[2024-11-04 22:47:19] Mean: -0.6577
[2024-11-04 22:47:19] Std: 0.2982
[2024-11-04 22:47:19] Processed video shape: (16, 256, 1088, 3)
[2024-11-04 22:47:19] Final range: [0, 255]
[2024-11-04 22:47:19] Saved frame 0 - Range: [0, 255], Mean: 35.11
[2024-11-04 22:47:19] Saved frame 1 - Range: [0, 255], Mean: 35.58
[2024-11-04 22:47:19] Saved frame 2 - Range: [0, 255], Mean: 36.43
[2024-11-04 22:47:19] Saved frame 3 - Range: [0, 255], Mean: 37.11
[2024-11-04 22:47:19] Saved frame 4 - Range: [0, 255], Mean: 37.34
[2024-11-04 22:47:19] Saved frame 5 - Range: [0, 255], Mean: 36.69
[2024-11-04 22:47:19] Saved frame 6 - Range: [0, 255], Mean: 36.67
[2024-11-04 22:47:19] Saved frame 7 - Range: [0, 255], Mean: 36.78
[2024-11-04 22:47:19] Saved frame 8 - Range: [0, 255], Mean: 37.60
[2024-11-04 22:47:19] Saved frame 9 - Range: [0, 255], Mean: 42.10
[2024-11-04 22:47:19] Saved frame 10 - Range: [0, 255], Mean: 48.02
[2024-11-04 22:47:19] Saved frame 11 - Range: [0, 255], Mean: 51.32
[2024-11-04 22:47:19] Saved frame 12 - Range: [0, 255], Mean: 53.30
[2024-11-04 22:47:19] Saved frame 13 - Range: [0, 255], Mean: 54.36
[2024-11-04 22:47:19] Saved frame 14 - Range: [0, 255], Mean: 53.96
[2024-11-04 22:47:19] Saved frame 15 - Range: [0, 255], Mean: 53.74
[2024-11-04 22:47:19] Successfully saved video to outputs/028/reference_epoch_000.mp4
[2024-11-04 22:47:19] Model training mode: True
[2024-11-04 22:47:19] EMA model training mode: False
[2024-11-04 22:47:19]
Latent size: [4, 16, 32, 32]
  0%|                                                                                                                                                 | 0/1000 [00:00<?, ?it/s]/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/checkpoint.py:92: UserWarning: None of the inputs have requires_grad=True. Gradients will be None 
  warnings.warn(
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [06:33<00:00,  2.54it/s]
[2024-11-04 22:53:52] 
=== Original Latents ===
[2024-11-04 22:53:52] Shape: torch.Size([1, 4, 16, 32, 32])
[2024-11-04 22:53:52] Range: [-1006.8341, 921.5656]
[2024-11-04 22:53:52] Mean: 0.1873
[2024-11-04 22:53:52] Std: 221.8859
[2024-11-04 22:53:52] 
=== Normalized Latents ===
[2024-11-04 22:53:52] Range: [-1.0047, 1.0047]
[2024-11-04 22:53:52] Mean: 0.0006
[2024-11-04 22:53:52] Std: 1.0000
[2024-11-04 22:53:53] 
=== Decoded Video ===
[2024-11-04 22:53:53] Shape: torch.Size([1, 3, 16, 256, 256])
[2024-11-04 22:53:53] Range: [-1.3613, 1.6924]
[2024-11-04 22:53:53] Mean: 0.0278
[2024-11-04 22:53:53] Std: 0.6782
[2024-11-04 22:53:53] 
=== Video Tensor Analysis ===
[2024-11-04 22:53:53] Shape: torch.Size([1, 3, 16, 256, 256])
[2024-11-04 22:53:53] Range: [-1.3613, 1.6924]
[2024-11-04 22:53:53] Mean: 0.0278
[2024-11-04 22:53:53] Std: 0.6782
[2024-11-04 22:53:53] Processed video shape: (16, 256, 256, 3)
[2024-11-04 22:53:53] Final range: [0, 255]
[2024-11-04 22:53:53] Saved frame 0 - Range: [6, 244], Mean: 114.67
[2024-11-04 22:53:53] Saved frame 1 - Range: [7, 239], Mean: 116.61
[2024-11-04 22:53:53] Saved frame 2 - Range: [8, 233], Mean: 115.22
[2024-11-04 22:53:53] Saved frame 3 - Range: [0, 248], Mean: 115.07
[2024-11-04 22:53:53] Saved frame 4 - Range: [6, 248], Mean: 114.80
[2024-11-04 22:53:53] Saved frame 5 - Range: [0, 239], Mean: 116.95
[2024-11-04 22:53:53] Saved frame 6 - Range: [7, 241], Mean: 114.96
[2024-11-04 22:53:53] Saved frame 7 - Range: [2, 245], Mean: 114.25
[2024-11-04 22:53:53] Saved frame 8 - Range: [1, 255], Mean: 116.00
[2024-11-04 22:53:53] Saved frame 9 - Range: [8, 244], Mean: 113.90
[2024-11-04 22:53:53] Saved frame 10 - Range: [6, 237], Mean: 115.52
[2024-11-04 22:53:53] Saved frame 11 - Range: [11, 241], Mean: 115.40
[2024-11-04 22:53:53] Saved frame 12 - Range: [1, 235], Mean: 117.85
[2024-11-04 22:53:53] Saved frame 13 - Range: [10, 244], Mean: 116.64
[2024-11-04 22:53:53] Saved frame 14 - Range: [10, 246], Mean: 114.99
[2024-11-04 22:53:53] Saved frame 15 - Range: [8, 232], Mean: 115.22
[2024-11-04 22:53:53] Successfully saved video to outputs/028/generated_epoch_000.mp4
Epoch 0:   0%|                                                                                                                                           | 0/5 [00:04<?, ?it/s]
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/Theseus-2/Train.py", line 702, in <module>
    main()
  File "/teamspace/studios/this_studio/Theseus-2/Train.py", line 627, in main
    batch = next(iter(dataloader))
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1344, in _next_data
    return self._process_data(data)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1370, in _process_data
    data.reraise()
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_utils.py", line 706, in reraise
    raise exception
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 309, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 317, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 155, in collate
    clone.update({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 155, in <dictcomp>
    clone.update({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 142, in collate
    return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 213, in collate_tensor_fn
    out = elem.new(storage).resize_(len(batch), *list(elem.size()))
RuntimeError: Trying to resize storage that is not resizable

⚡ ~/Theseus-2 
```

## Dataset

The model was trained on a small dataset consisting of 10 videos. The dataset was selected to demonstrate the basic functionality of the Theseus model on limited resources.

## License

This project is licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). For more information, please visit [this link](LICENSE.md).

---

Feel free to explore, modify, and experiment with the code. Contributions and improvements are welcome!

## Star History

<a href="https://star-history.com/#Rivera-ai/Theseus&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Rivera-ai/Theseus&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Rivera-ai/Theseus&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Rivera-ai/Theseus&type=Timeline" />
 </picture>
</a>
