<h1 align="center">M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation</h1>




<h4 align="center" style="line-height:1.4; margin-top:0.6rem">
  <a href="https://github.com/Graphic-Kiliani">Yiheng Zhang</a><sup>1*</sup>,
  <a href="https://www.caizhuojiang.com/">Zhuojiang Cai</a><sup>2*</sup>,
  <a href="https://openreview.net/profile?id=~Mingdao_Wang1">Mingdao Wang</a><sup>1*</sup>,
  <a href="https://openreview.net/profile?id=~Meitong_Guo1">Meitong Guo</a><sup>1</sup>,
  <a href="https://github.com/tingyunaiai9">Tianxiao Li</a><sup>1</sup>,
  <a href="https://xplorestaging.ieee.org/author/37088600614">Li Lin</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=KhFGpFIAAAAJ&hl=en">Yuwang Wang</a><sup>1†</sup>
</h4>

<p align="center" style="margin:0.2rem 0 0.6rem 0;">
  <sup>1</sup> Tsinghua University &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>2</sup> Beihang University &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>3</sup> Migu Beijing Research Institute
</p>

<p align="center" style="font-size:0.95em; color:#666; margin-top:0;">
  * Equal contribution &nbsp;&nbsp;|&nbsp;&nbsp; † Corresponding author
</p>

<p align="center">
  <a href="https://graphic-kiliani.github.io/M3DLayout/">
    <img src="https://img.shields.io/badge/Project%20Page-blue.svg" alt="Project Page" height="22">
  </a>
  <a href="https://arxiv.org/abs/2509.23728">
      <img src="https://img.shields.io/badge/arXiv-b31b1b.svg?logo=arXiv&logoColor=white" alt="arXiv height="22">
  </a>
</p>



<p align="center">
    <img width="90%" alt="pipeline", src="./assets/Teaser.png">
</p>
</h4>


We have released our **Object Retrieval** and **Rendering** code, come and try it!!!
Other codes and datset will be released as soon as possible.
## Abstract

In text-driven 3D scene generation, object layout serves as a crucial intermediate representation that bridges high-level language instructions with detailed geometric output. It not only provides a structural blueprint for ensuring physical plausibility but also supports semantic controllability and interactive editing. 

However, the learning capabilities of current 3D indoor layout generation models are constrained by the limited scale, diversity, and annotation quality of existing datasets. To address this, we introduce **M3DLayout, a large-scale, multi-source dataset for 3D indoor layout generation**. M3DLayout comprises **21,367 layouts** and over **433k object instances**, integrating three distinct sources: real-world scans, professional CAD designs, and procedurally generated scenes. Each layout is paired with detailed structured text describing global scene summaries, relational placements of large furniture, and fine-grained arrangements of smaller items. This diverse and richly annotated resource enables models to learn complex spatial and semantic patterns across a wide variety of indoor environments. 

To assess the potential of M3DLayout, we establish a benchmark using a text-conditioned diffusion model. Experimental results demonstrate that our dataset provides a solid foundation for training layout generation models. Its multi-source composition enhances diversity, notably through the Inf3DLayout subset which provides rich small-object information, enabling the generation of more complex and detailed scenes. We hope that M3DLayout can serve as a valuable resource for advancing research in text-driven 3D scene synthesis.


## TODO
- [x] Release Object Retrieval code of M3DLayout
- [x] Release rendering code of layouts and scenes
- [ ] Release inference code of M3DLayout
- [ ] Provide training instruction for M3DLayout
- [ ] Release M3DLayout dataset


## Citation
If you find our work helpful, please consider citing:
```bibtex
@article{zhang2025m3dlayout,
      title={M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation}, 
      author={Yiheng, Zhang and Zhuojiang, Cai and Mingdao, Wang and Meitong, Guo and Tianxiao, Li and Li, Lin and Yuwang, Wang},
      journal={arXiv preprint arXiv:2509.23728},
      year={2025},
      url={https://arxiv.org/abs/2509.23728}, 
}
```