# fastCell
**fastCell** is a free open-source python software package designed to enable biologists without training in computer vision or programming to utilize cutting edge deep learning technology in their quantitative analysis of biological images.

**fastCell** is developed at the [Mouse Imaging Centre](https://github.com/Mouse-Imaging-Centre), where the acquisition of increasingly ambiguous cell types proved too challenging for [CellProfiler](https://cellprofiler.org)'s conventional image analysis algorithms.

Dr. Dulcie Vousden originally published results using CellProfiler in her paper [Whole-brain mapping of behaviourally induced neural activation in mice](https://www.ncbi.nlm.nih.gov/pubmed/24760545). [tvPipeline]() now automates the entire workflow including the useage of **fastCell**.

# Installation
**NB:** fastCell is built on [fastai](https://github.com/fastai/fastai) which also requires a specific version of [PyTorch](https://github.com/pytorch/pytorch) that may differ depending on your system. As such, it is highly recommended that the user **first installs `fastai-1.0.5x` and its dependencies, then fastCell into a [virtual environment](https://docs.python.org/3.7/tutorial/venv.html)**.

## Developer Install
Python's setuptools [doesn't work with opencv-python](https://github.com/skvark/opencv-python/issues/47#issuecomment-332830074), so install it with pip first. 
```bash
pip install opencv-python==4.1.0.25
```

After installing fastai and opencv-python, install fastCell.
```bash
git clone https://github.com/Mouse-Imaging-Centre/fastCell.git
cd fastCell
python3 setup.py develop
```