本深度学习框架集成了1000+模型（包含100+原创的CNN-Transformer混合架构，适用于2D/3D多模态图像处理）。
该框架采用模块化设计，各功能组件可独立配置：训练策略优化、数据预处理、模块结构重组以及可视化分析（支持热力图与梯度图生成），
从而显著提升模型开发全流程（训练/验证/测试）的效率。






python 14classification_YXY_查看网络参数量.py >>visual/model_visual.txt
这样运行的时候数据就保持到文件中了

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

del 后面要加一句torch.cuda.empty_cache()

python 14classification_YXY_查看网络参数量.py --gpu 0,1 --batchsize 32 --reduction 16 >>visual/model_visual.txt
python 15classification_YXY_56x56.py --gpu 0,1 --batchsize 32 --reduction 16 >>visual/model_visual.txt


# 这里对densenet进行训练，numwork使用4个
python 18classification_YXY_CE56.py --gpu 0,1 --batchsize 64 --reduction 16 >>visual/model_visual.txt

python 18classification_YXY_CE56.py --gpu 0,1 --batchsize 64 --reduction 16 >>visual/model_visual.txt
python 19classification_YXY_ChestXray2017.py --gpu 0,1 --batchsize 16 --reduction 16 >>visual/model_visual.txt
python 20classification_YXY_CT56.py --gpu 0,1 --batchsize 16 --reduction 16 >>visual/model_visual.txt
python 22classification_YXY_ChestXray14.py --gpu 0,1 --batchsize 16 --reduction 16 >>visual/model_visual.txt
python 24ChestXray2017.py --gpu 0,1 --batchsize 28 --reduction 16 >>visual/model_visual.txt
python 25COVID_CT.py --gpu 0,1 --batchsize 16 --reduction 16 >>visual/model_visual.txt
python 27PET_CT.py --gpu 0,1 --batchsize 28 --reduction 16 >>visual/model_visual.txt
python 28Chest_CT.py --gpu 0 --batchsize 28 --reduction 16 >>visual/model_visual.txt
python 29Chest_CT_D.py --gpu 0 --batchsize 28 --reduction 16 >>visual/model_visual.txt
python 29Chest_CT_D.py --gpu 0 --batchsize 28
python 31rep_COVID.py --gpu 0,1 --batchsize 48 >>visual/model_visual.txt
python 31rep_COVID.py --gpu 0 --batchsize 16 >>visual/model_visual.txt


``

python 16classification_YXY_56x56_224x224.py --gpu 0,1 --batchsize 32 --reduction 16 >>model_visual.txt
python 15classification_YXY_56x56.py --gpu 0,1 --batchsize 32 --reduction 16 >>model_visual.txt

nvidia-smi
查看显存后查看PID
set CUDA_VISIBLE_DEVICES=0

```shell·       CAM 
# 实际上这里的--weight-path 参数并没有使用上
python Grad_CAM/02Grad_CAM_pytorch/main.py --image-path Grad_CAM/02Grad_CAM_pytorch/examples/pic1.jpg --model-index 114 --weight-path /home/yxy/weights/alexnet.pth
# 这里默认是使用一张新冠肺炎的人的图像，
python Grad_CAM/02Grad_CAM_pytorch/main.py --model-index 114 --weight-path /home/yxy/weights/alexnet.pth
```