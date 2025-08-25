The code will be uploaded after the paper is accepted.

-----------------------------------------------------------

Due to page limitations in the submitted manuscript, we have omitted some experimental results and provide them here as a supplement. These contents will be incorporated into the manuscript in the revised version.

-----------------------------------------------------------
## 1. TLG Multi-Object Segmentation Visualization

<div align="center">
  <img width="80%" src="Figs/supp_1.png">
  <p><b>Figure 1.</b> Multi-Objective Qualitative Visualization Analysis: Visualizing Segmentation Results under a 1-Shot Setting on the COCO-20i Datasets. Each pair of columns corresponds to a fold in the meta-learning paradigm.</p>
</div>

This section provides a systematic evaluation of the TLG model's performance in multi-object segmentation scenarios. Compared to single-object segmentation, multi-object segmentation places higher demands on the model's contextual understanding. This is primarily due to the inherent attention bias of convolutional neural networks, which tend to prioritize salient object features in the image. This characteristic not only tests the model's ability to integrate multi-scale spatial information but also presents a significant challenge in modeling the interactions between objects in complex scenes. 

As illustrated in Figure 1, we systematically evaluate TLG's multi-object segmentation performance across different folds of the COCO-20<sup>i</sup> datasets. In the first column (left to right) depicting the *elephant* category, TLG demonstrates robust anti-occlusion capabilities by accurately segmenting both primary targets and partially obscured contours. In the third column, under the *umbrella* category, multiple umbrellas diminish in size from foreground to background, exhibiting significant spatial depth. TLG adeptly segments these smaller targets, demonstrating its robust spatial modeling capabilities. Similarly, in the fifth column featuring the *sheep* category, the lower portion of the image is dimly lit, obscuring object details. Nonetheless, TLG accurately segments multiple sheep, highlighting its proficiency in multi-scale feature fusion and semantic understanding. Finally, when evaluating densely stacked targets such as *orange* and *carrot*, TLG maintains clarity and precision in segmentation, underscoring its strong robustness.

## 2. Limitations

<div align="center">
  <img width="80%" src="Figs/supp_2.png">
  <p><b>Figure 2.</b> Visual Analysis of TLG Segmentation Limitations.</p>
</div>

In this section, we present a visual analysis of the segmentation limitations of TLG. As shown in Fig. 2, these limitations can be categorized into three aspects: (1) segmentation under low-light conditions. For instance, in the first row corresponding to the “Person” category, the dim background within the train and the subject’s dark attire cause TLG to fail in segmenting the object accurately. Therefore, TLG may exhibit limitations in nighttime scenarios. (2) Over-segmentation of fine-grained objects. Images from daily life scenes are frequently characterized by low resolution and blurriness. For example, the “bicycle” category, which features a slender and complex structure with numerous components such as tires and chains, may lead the model to over-respond to local details, resulting in over-segmentation. To address this issue, future work could explore the integration of super-resolution mechanisms to improve the segmentation performance of fine-grained objects. (3) Under-segmentation caused by indistinct local features. For target categories with partially visible and low-texture appearances, TLG may exhibit under-segmentation in certain samples. For instance, in the case of the “diningtable” category, when only a portion of the tabletop is visible and the surface lacks prominent decorations or texture cues, the model may fail to capture its full contour, resulting in incomplete segmentation.
