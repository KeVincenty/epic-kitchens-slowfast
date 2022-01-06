HOME_DIR=/home/yinyuan/workspace/epic-kitchens-slowfast
cd $HOME_DIR
ps -ef | grep 'python -c from multiprocessing.spawn' | grep -v grep | awk '{print $2}' | xargs kill
export CUDA_VISIBLE_DEVICES=4,5,6,7
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml