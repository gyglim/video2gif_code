================================================================================

## Video2GIF code, version 0.9

This repository provides the pretrained model used in our Video2GIF work.


For information on how to use the code, please see our [Tutorial](./video2gif_tutorial.ipynb).

In order to run the tutorial, you will first need to install the video2gif package by running
```
    python setup.py install --user
```

in the code directory.

If you don't have cuDNN > 4.0 available, you need to use the Lasagne fork that provides 3D convolutions and pooling implementations without cuDNN. You can get it from https://github.com/gyglim/Lasagne

Note: In case the data is not available on the ETH links, where's a backup on drive:

C3D weights: https://drive.google.com/file/d/12J5Rsz3zBp3gKKvrmRU3B8s9OTAcfR7C/
Snippet mean: https://drive.google.com/file/d/1oKN5APlKzf70X4l4K4CIcuOUfbUCdHUt/
Video2GIF weights: https://drive.google.com/file/d/1Qvd6xXucwiis0bqlcoHB6pMIh7GXgdvK/
Example video: https://drive.google.com/file/d/17LdWCqUBy0-4uXLxbw9RSUH_fGTQAbVQ/

================================================================================

If you end up using the code, we ask you to cite the following paper:

    Michael Gygli, Yale Song, Liangliang Cao
    "Video2GIF: Automatic Generation of Animated GIFs from Video,"
    IEEE CVPR 2016

If you have any question regarding the dataset, please contact:

    Michael Gygli <gygli@vision.ee.ethz.ch>

License: This code is licensed under GPL version 3, see LICENSE file

### Note: There is a patent pending on the ideas presented in this work so this code should only be used for academic purposes.

   
Last edit: June 21, 2016
