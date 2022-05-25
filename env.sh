conda create --name DLRNet python=3.7 -y 
conda activate DLRNet
conda install numpy pandas tqdm matplotlib scikit-learn -y 
conda install pytorch=1.10.2 ignite torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install wandb