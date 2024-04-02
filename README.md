# **Language Grounded Single Source Domain Generalization in Medical Image Segmentation [ISBI 2024]** 
[Shahina Kunhimon](https://github.com/ShahinaKK),
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Salman Khan](https://salman-h-khan.github.io),
and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en) 
![](https://i.imgur.com/waxVImv.png)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.01272)

<table style="width: 100%;">
  <tr>
    <td style="width: 50%; vertical-align: top;">
      <img src="https://github.com/ShahinaKK/LG_SDG/blob/main/mainfig.png" alt="main figure" style="width: 100%;">
    </td>
    <td style="width: 50%; vertical-align: top;">
      <b>Abstract</b>
        <p>SDG holds promise for more reliable and consistent image segmentation across real-world clinical settings particularly in the medical domain, where data privacy and acquisition cost constraints often limit the availability of diverse datasets. Textual cues describing the anatomical structures, their appearances, and variations across various imaging modalities can guide the model in domain adaptation, ultimately contributing to more robust and consistent segmentation. In this paper, we propose an approach that explicitly leverages textual information by incorporating a contrastive learning mechanism guided by the text encoder features to learn a more robust feature representation. We assess the effectiveness of our text-guided contrastive feature alignment technique in various scenarios, including cross-modality, cross-sequence, and cross-site settings for different segmentation tasks. </p>
      </td>
  </tr>
</table>



## Installation
<details>
  <summary>
    <b> Create and activate conda environment.</b>
  </summary>
    <pre>
    conda env create -f lgsdg.yml
    conda activate lgsdg</pre>
  </code>
</details>

<details>
  <summary>
    <b> Run the setup file for CCSDG module.</b>
  </summary>
    <pre>
    cd CCSDG
    pip install -e. </pre>
  </code>
</details>

## Download Dataset and Text Embeddings
<details>
  <summary>
    <b> For Fundus dataset </b>
  </summary>
Download the <a href="https://zenodo.org/record/6325549">CCSDG Fundus dataset</a>.
</details>
<details>
  <summary>
    <b> For Abdominal and Cardiac datasets </b>
  </summary>
Download the <a href="https://drive.google.com/file/d/1WlXGt3Nffzu1bn6co-qaidHjqWH51smU/view?usp=share_link">SLAUG Processed datasets</a> and follow the instructions in this repo (<a href="https://github.com/Kaiseem/SLAug">SLAug</a>) to organize the data.
</details>
<details>
<summary>
<b> Get the Text Embeddings</b>
  </summary>
Download the <a href="https://drive.google.com/file/d/1_GsjcX7huV85BdMBS3ojI06C1YKps_Gg/view?usp=drive_link">Text_Embeddings</a> and unzip it to use them directly. 
<b> OR</b>
You can download the jupyter notebooks from <a href="https://drive.google.com/file/d/1CYl8ZzndL06xjpFN04K0rm8ZDZDDz4pW/view?usp=drive_link">Notebooks</a>, unzip it, update the text annotations and generate the text embeddings.
</details>

## Inference using Pretrained Models
<details>
  <summary>
    <b> Fundus Dataset</b>
  </summary>
    <p>Download the <a href="https://drive.google.com/file/d/1ISuJIVkXlIsZndzP9F5AcpiA6Rm69IQv/view?usp=drive_link">pretrained model</a> weights and put it in the directory path:</p> 
    <pre>OUTPUT_FOLDER/unet_ccsdg_source_Magrabia/checkpoints/ </pre>
   <p>To run the inference:</p> 
  <pre>
   cd CCSDG
   python ccsdg/inference/run_inference.py --model unet_ccsdg --gpu 0 --tag source_Magrabia --log_folder OUTPUT_FOLDER -r ./CCSDG_DATA --ts_csv ./CCSDG_DATA/MESSIDOR_Base1_test.csv</pre>
   
  </code>
</details>
<details>
  <summary>
    <b> For Abdominal and Cardiac datasets</b>
  </summary>
    <p>Download the <a href="https://drive.google.com/file/d/15dPz675sNr9ecgbvLlxA6ai8MfRNwFFn/view?usp=drive_link">pretrained models</a> and run the inference:</p> 
  <pre>
   cd SLAug
   python test.py -r $CHECKPOINT</pre>
   
  </code>
</details>

## Training the Models
<details>
  <summary>
    <b> Fundus Dataset</b>
  </summary>
    <p>
    Update the paths and run the bash script:</p>
  <pre>
   cd CCSDG
   bash train.sh</pre>
  </code>
</details>

<details>
  <summary>
    <b> Abdominal Dataset</b>
  </summary>
    <p>For CT -&gt; MRI:</p>
    <pre>
      cd SLAug
      python main.py --base configs/efficientUnet_SABSCT_to_CHAOS.yaml --seed 23</pre>
    <p>For MRI -&gt; CT:</p>
    <pre>
    cd SLAug
    python main.py --base configs/efficientUnet_CHAOS_to_SABSCT.yaml --seed 23</pre>
  </code>
</details>

<details>
  <summary>
    <b> Cardiac Dataset</b>
  </summary>
    <p>For bSSFP -&gt; LEG:</p>
    <pre>
    cd SLAug
    python main.py --base configs/efficientUnet_bSSFP_to_LEG.yaml --seed 23</pre>
    <p>For LEG -&gt; bSSFP:</p>
    <pre>
    cd SLAug
    python main.py --base configs/efficientUnet_LEG_to_bSSFP.yaml --seed 23</pre>
  </code>
</details>

## Contact
Should you have any questions, please create an issue in this repository or contact shahina.kunhimon@mbzuai.ac.ae

## References
Our code is build on the repositories of [SLAug](https://github.com/Kaiseem/SLAug) and [CCSDG](https://github.com/ShishuaiHu/CCSDG). We thank them for releasing their code.

<hr>
