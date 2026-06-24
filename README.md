# Through the Looking Glass: A Dual Perspective on Weakly-Supervised Few-Shot Segmentation

<p align="center">
  <a href="https://arxiv.org/abs/2508.16159"><img src="https://img.shields.io/badge/arXiv-2605.19623-b31b1b.svg" alt="arXiv"></a>
</p>



<p align="center">
  <a href="https://scholar.google.com/citations?user=VtduT6sAAAAJ&hl=zh-CN">Jiaqi Ma</a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://scholar.google.com.hk/citations?user=LKaWa9gAAAAJ&hl=zh-CN&oi=ao">Guo-Sen Xie</a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=4C7mvOwAAAAJ">Fang Zhao</a><sup>2</sup> &nbsp;·&nbsp;
  <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=L6J2V3sAAAAJ">Zechao Li</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Nanjing University of Science and Technology &nbsp;&nbsp;
  <sup>2</sup> Nanjing University
</p>

## :loudspeaker: Notice
If you like this project, please ⭐ it on GitHub, Thanks!

## :fire: News

[June 24, 2026]： We have released all the code of TLG.

[June 22, 2026]： AFANet is accepted by IEEE Transactions on Image Processing (TIP).

## Abstract
<div align=center><img width="80%" src="assets/fig.2.png"></div> 

Few-shot learning aims to recognize novel concepts by leveraging prior knowledge learned from a few samples. However, for visually intensive tasks such as few-shot semantic segmentation, pixel-level annotations are time-consuming and costly. Therefore, in this work, we utilize the more challenging image-level annotations and propose an adaptive frequency-aware network (AFANet) for weakly-supervised few-shot semantic segmentation (WFSS). Specifically, we first propose a cross-granularity frequency-aware module (CFM) that decouples RGB images into high-frequency and low-frequency distributions and further optimizes semantic structural information by realigning them. Unlike most existing WFSS methods using the textual information from the language-vision model CLIP in an offline learning manner, we further propose a CLIP-guided spatial-adapter module (CSM), which performs spatial domain adaptive transformation on textual information through online learning, thus providing cross-modal semantic information for CFM. Extensive experiments on the Pascal-5 and COCO-20 datasets demonstrate that AFANet has achieved state-of-the-art performance.

## Experiments
<div align=center><img width="80%" src="assets/voc_result.png"></div> 
