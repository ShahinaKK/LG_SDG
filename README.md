# **Language Grounded Single Source Domain Generalization in Medical Image Segmentation [ISBI 2024]** 
[Shahina Kunhimon](https://github.com/ShahinaKK),
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Salman Khan](https://salman-h-khan.github.io),
and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en) 
![](https://i.imgur.com/waxVImv.png)
<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Single source domain generalization (SDG) 
%aims at developing models that can effectively adapt to unseen domains using only a single source dataset. It
holds promise for more reliable and consistent image segmentation across real-world clinical settings particularly in the medical domain, where data privacy and acquisition cost constraints often limit the availability of diverse datasets. Depending solely on visual features hampers the model's capacity to adapt effectively to various domains, primarily because of the presence of spurious correlations and domain-specific characteristics embedded within the image features. 
 Incorporating text features alongside visual features is a potential solution to enhance the model's understanding of the data, as it goes beyond pixel-level information to provide valuable context. Textual cues describing the anatomical structures, their appearances, and variations across various imaging modalities can guide the model in domain adaptation, ultimately contributing to more robust and consistent segmentation. In this paper, we propose an approach that explicitly leverages textual information by incorporating a contrastive learning mechanism guided by the text encoder features to learn a more robust feature representation.  We assess the effectiveness of our text-guided contrastive feature alignment technique in various scenarios, including cross-modality, cross-sequence, and cross-site settings for different segmentation tasks. Our approach achieves favorable performance against existing methods in literature.
</details>

## Installation
1. Create and activate conda environment.
   ```shell
    conda env create -f lgsdg.yml
    conda activate lgsdg
2. Run the setup file for CCSDG module.
      ```shell
    cd CCSDG
    pip install -e.

## Dataset 
<details>
  <summary>
    <b>1) For Fundus dataset </b>
  </summary>
Download the [CCSDG Fundus dataset](https://zenodo.org/record/6325549) and unzip it.
</details>
<details>
  <summary>
    <b>1)For Abdominal and Cardiac datasets </b>
  </summary>
Download the [SLAUG Processed datasets](https://drive.google.com/file/d/1WlXGt3Nffzu1bn6co-qaidHjqWH51smU/view?usp=share_link) and follow the instructions in this repo (https://github.com/Kaiseem/SLAug) to organize the data.
</details>
