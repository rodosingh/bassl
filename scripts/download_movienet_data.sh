#!/bin/bash
#SBATCH -A rodosingh
#SBATCH -c 2
#SBATCH --gres=gpu:0
#SBATCH -w gnode060
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home2/rodosingh/PROJECT/bassl/logs/anno.log
#SBATCH --mail-user aditya.si@research.iiit.ac.in
#SBATCH --mail-type ALL

# DATA_DIR=./bassl/data/movienet
# mkdir -p ${DATA_DIR}/anno
# mkdir -p ${DATA_DIR}/240P_frames

# # download key-frames of shots (requires almost 160G)
# wget -N -P ${DATA_DIR} https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/movienet/movie1K.keyframes.240p.v1.zip

# Recently OpenMMlab has changed the link of the dataset.
# They moved it to OpenDataLab site. And the dataset can be found at: https://opendatalab.com/MovieNet/
# To download it in CLI, you need to register and login to the site.
# Then, you can download the dataset by using the following command:

# Configure install
# pip install opendatalab

# # Upgraded version
# pip install -U opendatalab

# odl login                  # Login
# odl info   MovieNet        # View dataset metadata
# odl ls     MovieNet        # View a list of dataset files
# mkdir -p /ssd_scratch/cvit/rodosingh/data/
# cd /ssd_scratch/cvit/rodosingh/data/
# source $HOME/virtualenvs/BAS/bin/activate
# yes | odl get    MovieNet        # Download this dataset
# echo "Downloaded the dataset"


# # download annotations for data loader (requires almost 200M)
# wget -N -P ${DATA_DIR} https://arena.kakaocdn.net/brainrepo/bassl/data/anno.tar
mkdir -p /scratch/cvit/rodosingh/data/MovieNet/raw/
cd /scratch/cvit/rodosingh/data/MovieNet/raw/
wget https://arena.kakaocdn.net/brainrepo/bassl/data/anno.tar


# # decompress
# unzip ${DATA_DIR}/movie1K.keyframes.240p.v1.zip -d ${DATA_DIR}
# for FILE in `ls ${DATA_DIR}/240P -1`; do tar -xvf "${DATA_DIR}/240P/${FILE}" -C  ${DATA_DIR}/240P_frames ; done
# tar -xvf ${DATA_DIR}/anno.tar -C ${DATA_DIR}/anno
