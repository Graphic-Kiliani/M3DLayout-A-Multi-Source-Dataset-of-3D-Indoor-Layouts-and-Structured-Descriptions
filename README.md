<h1 align="center">M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation</h1>


<h4 align="center">

[Yiheng Zhang](https://github.com/Graphic-Kiliani), [Zhuojiang Cai](https://www.caizhuojiang.com/), [Mingdao Wang](https://openreview.net/profile?id=~Mingdao_Wang1), [Meitong Guo](https://openreview.net/profile?id=~Meitong_Guo1), [Tianxiao Li](https://github.com/tingyunaiai9), [Li Lin](https://xplorestaging.ieee.org/author/37088600614), [Yuwang Wang](https://scholar.google.com/citations?user=KhFGpFIAAAAJ&hl=en)

Tsinghua University, Beihang University, Migu Beijing Research Institute 

[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://graphic-kiliani.github.io/M3DLayout/)


<p align="center">
    <img width="90%" alt="pipeline", src="./assets/Teaser.png">
</p>
</h4>

## Abstract

In text-driven 3D scene generation, object layout serves as a crucial intermediate representation that bridges high-level language instructions with detailed geometric output. It not only provides a structural blueprint for ensuring physical plausibility but also supports semantic controllability and interactive editing. 

However, the learning capabilities of current 3D indoor layout generation models are constrained by the limited scale, diversity, and annotation quality of existing datasets. To address this, we introduce **M3DLayout, a large-scale, multi-source dataset for 3D indoor layout generation**. M3DLayout comprises **15,080 layouts** and over **258k object instances**, integrating three distinct sources: real-world scans, professional CAD designs, and procedurally generated scenes. Each layout is paired with detailed structured text describing global scene summaries, relational placements of large furniture, and fine-grained arrangements of smaller items. This diverse and richly annotated resource enables models to learn complex spatial and semantic patterns across a wide variety of indoor environments. 

To assess the potential of M3DLayout, we establish a benchmark using a text-conditioned diffusion model. Experimental results demonstrate that our dataset provides a solid foundation for training layout generation models. Its multi-source composition enhances diversity, notably through the Inf3DLayout subset which provides rich small-object information, enabling the generation of more complex and detailed scenes. We hope that M3DLayout can serve as a valuable resource for advancing research in text-driven 3D scene synthesis.


## TODO
- [ ] Provide inference code of M3DLayout
- [ ] Provide training instruction for M3DLayout
- [ ] Release M3DLayout dataset


## Citation
If you find our work helpful, please consider citing:
```bibtex
@article{zhang2025m3dlayout,
  author    = {Zhang, Yiheng and Cai, Zhuojiang, and Wang, Mingdao, and Guo, Meitong, and Li, Tianxiao, and Lin, Li, and  Wang, Yuwang},
  journal   = {arXiv preprint arXiv:},
  year      = {2025},
}
```