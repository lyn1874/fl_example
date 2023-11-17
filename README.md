# Use the DTU hpc center 

- log in
```ssh user_name@login1.gbar.dtu.dk```

- go to interactive node
  - ```linuxsh``` node without gpu
  - ```sxm2sh``` node with gpu
  - ```nvidia-smi``` check the existing gpus
- Load the necessary packages, e.g.,
  ```
  module load scipy/1.10.1-python-3.9.17 matplotlib/3.7.1-numpy-1.24.3-python-3.9.17
  ```
  Note you may need to load other packages based on your own requirements, please check available packages via `module avail ...` 
 
- In the interactive node, create virtual env (only run the code below once!)
  ```
  python3 -m venv torch_dl
  source torch_dl/bin/activate
  python -m pip install torch torchvision
  ```
  Note, again, you may need to install other required packages in this virtual env, e.g., seaborn... via `python -m pip install .... `
- After this, check if you can get the output below:
  ![](hpc_image.png)
- Run the cifar10 experiment (this is just an example, you need to change some hyperparameters, i.e., number of rounds, number of clients, gpu index
  ```
  ./run_cifar.sh
  ```
  - If you get a segmentation fault in an interactive node, check if you are using the correct GPU, i.e., do nvidia-smi, see which gpu is available, assign export CUDA_VISIBLE_DEVICES=?. I also had a segmentation fault once as I was using a too-large batch size. Reducing the batch size solved the problem. Of course, other things may also cause this fault. 
  - If you submit a job script and only require a single gpu, then the gpu index should be 0.
 
- If you want to submit the job, do
  ```bsub < submit_job.sh```


