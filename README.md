# Theseus: A Basic T2V Model Inspired by CogVideoX

This repository contains a very basic implementation of the ideas presented in the CogVideoX paper, including 3D attention mechanisms and causal convolutional layers. The results shown were obtained after training for 100 epochs on a mini-dataset of 10 videos using a NVIDIA T4 GPU on Google Colab.

## Overview

Theseus is a simplified Text-to-Video (T2V) model that leverages core concepts from the CogVideoX architecture. While not a complete implementation, it provides a foundational framework to explore the intersection of text and video modalities through deep learning.

Key features:

- **3D Convolutions:** Utilized for video encoding and decoding to capture spatial and temporal information.
- **Causal Convolutions:** Applied to ensure temporal consistency in video generation.
- **Transformer Integration:** Combines video and text embeddings to generate coherent video outputs based on textual descriptions.

## Results

Below are sample videos generated by Theseus after 100 epochs of training:

### Video 1

<video width="320" height="240" controls>
  <source src="reconstructed_video_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Video 2

<video width="320" height="240" controls>
  <source src="reconstructed_video_2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Video 3

<video width="320" height="240" controls>
  <source src="reconstructed_video_3.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

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
