
pathgt=/home/vislab/jieli23/dataset/LSUN/churches_val


# --
pathpd=/home/vislab/jieli23/proj/latent-diffusion/logs/lsun_churches256/samples/00500000/2023-11-15-20-21-20/img
python fid_score.py ${pathgt} ${pathpd} --batch-size 50  --gpu 0 --dims 2048

# --
pathpd=/home/vislab/jieli23/proj/latent-diffusion/logs/lsun_churches256/samples/00500000/2023-11-15-20-42-15/img
python fid_score.py ${pathgt} ${pathpd} --batch-size 50  --gpu 0 --dims 2048

# --
pathpd=/home/vislab/jieli23/proj/latent-diffusion/logs/lsun_churches256/samples/00500000/2023-11-15-20-52-03/img
python fid_score.py ${pathgt} ${pathpd} --batch-size 50  --gpu 0 --dims 2048

# --
pathpd=/home/vislab/jieli23/proj/latent-diffusion/logs/lsun_churches256/samples/00500000/2023-11-15-20-55-45/img
python fid_score.py ${pathgt} ${pathpd} --batch-size 50  --gpu 0 --dims 2048

# --
pathpd=/home/vislab/jieli23/proj/latent-diffusion/logs/lsun_churches256/samples/00500000/2023-11-15-21-08-20/img
python fid_score.py ${pathgt} ${pathpd} --batch-size 50  --gpu 0 --dims 2048

# --
pathpd=/home/vislab/jieli23/proj/latent-diffusion/logs/lsun_churches256/samples/00500000/2023-11-17-00-26-21/img
python fid_score.py ${pathgt} ${pathpd} --batch-size 50  --gpu 0 --dims 2048