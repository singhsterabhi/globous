git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.1.0
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" -DOPENCV_EXTRA_MODULES_PATH=~/Tools/opencv_contrib/modules ..
make -j4
sudo make install
