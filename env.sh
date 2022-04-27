conda create --name PDE python=3.7 -y 
conda activate PDE
conda install numpy pandas tqdm matplotlib scikit-learn -y 
conda install pytorch=1.10.2 ignite torchvision torchaudio cudatoolkit=11.3 -c pytorch 
pip install wandb