from setuptools import setup, find_namespace_packages

setup(name='ccsdg',
      packages=find_namespace_packages(include=["ccsdg", "ccsdg.*"]),
      version='0.0.1',
      description='Channel-level Contrastive Single Domain Generalization for Medical Image Segmentation',
      author='Shishuai Hu',
      author_email='sshu@mail.nwpu.edu.cn',
      license='MIT License',
      install_requires=[
          "torch>=1.6.0a",
          "torchvision",
          "tqdm",
          "dicom2nifti",
          "scikit-image>=0.14",
          "medpy",
          "scipy",
          "batchgenerators>=0.21",
          "numpy",
          #"sklearn",
          "SimpleITK",
          "pandas",
          "requests",
          "hiddenlayer", "graphviz", "IPython",
          "nibabel", 'tifffile',
          "tensorboard"
      ],
      entry_points={
          'console_scripts': [
              'ccsdg_train = ccsdg.training.run_training:main',
              'ccsdg_test = ccsdg.inference.run_inference:main',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'single domain generalization', 'contrastive learning']
      )
