# fastCell
fastCell is a free open-source python software package designed to enable biologists without training in computer vision or programming to utilize cutting edge deep learning technology in their quantitative analysis of biological images.

fastCell is developed at the [Mouse Imaging Centre](https://github.com/Mouse-Imaging-Centre), where the acquisition of increasingly ambiguous cell types proved too challenging for [CellProfiler](https://cellprofiler.org)'s conventional image analysis algorithms.

Dr. Dulcie Vousden originally published results using CellProfiler in her paper [Whole-brain mapping of behaviourally induced neural activation in mice](https://www.ncbi.nlm.nih.gov/pubmed/24760545). [tvPipeline]() now automates the entire workflow including the useage of **fastCell**.

# Installation
**NB:** fastCell is built on fastai which also requires a specific version of PyTorch that may differ depending on your system.

As such, it is highly recommended that you **first [install fastai](https://docs.fast.ai/install.html)** and its dependencies, then fastCell, **all into a [virtual environment](https://docs.python.org/3.7/tutorial/venv.html).**

## Developer Install
Python's setuptools [doesn't work with opencv-python](https://github.com/skvark/opencv-python/issues/47#issuecomment-332830074), so install it with pip first. 
```bash
pip3 install opencv-python==4.1.0.25
```

After installing fastai and opencv-python, install fastCell.
```bash
git clone https://github.com/Mouse-Imaging-Centre/fastCell.git
cd fastCell
python3 setup.py install
```

# Installation Issues
If you encounter problems with pip, please make sure you have the latest pip:

`pip3 install --upgrade pip`

For all fastai related issues, please follow [their instructions](https://github.com/fastai/fastai#installation-issues).

If you are trying to _install_ fastCell on ComputeCanada's Graham or Cedar clusters, you don't need to as they are already installed. If you truly want your own installation on Graham/Cedar, comment out the opencv-python dependency from setup.py  and load the system's `opencv/4.1.0` module. ComputeCanada has crippled its pip to ignore binary "manylinux" wheels as they typically do not work or underperform so they typically build python wheels from source or install it witrh a module when it's not practical. The crippling file is `/cvmfs/soft.computecanada.ca/custom/python/site-packages/_manylinux.py`