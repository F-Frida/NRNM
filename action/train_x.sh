# CUDA_VISIBLE_DEVICES=0 python3 action_mylstm_cv.py --nlayers=3 --my 0 1 2 --data_mode=CV --lr=0.001 --wd=0.0005 --stride=2 --bl=4 --nhid=256 --sb=cv_s2b4_3layer_z2_wd001_mix012_256_0314
# CUDA_VISIBLE_DEVICES=0 python3 action_mylstm_cv.py --nlayers=3 --order=4 --cell_type=HO --clip=0.5 --my 0 1 2 --data_mode=CV --lr=0.0001 --wd=0.0005 --stride=2 --bl=4 --nhid=256 --sb=cv_ho4rnn_256_0317
# CUDA_VISIBLE_DEVICES=0 python3 action_mylstm.py --nlayers=3 --model=GRU --cell_type=rnn  --clip=0.5 --my 0 1 2 --lr=0.0001 --wd=0.0005 --stride=2 --bl=4 --nhid=256 --sb=GRU_256_0318
# CUDA_VISIBLE_DEVICES=0 python3 action_mylstm_ae_cv.py --nlayers=3 --my 0 1 --data_mode=CV --lr=0.001 --wd=0.001 --stride=4 --bl=8 --nhid=256 --sb=cv_s4b8_3layer_z2_wd001_mix01_256_aezout_0313
# CUDA_VISIBLE_DEVICES=0 python3 action_mylstm.py --nlayers=3 --my 0 1  --lr=0.001 --wd=0.0001 --stride=3 --bl=12 --nhid=128 --sb=s3b12_3layer_z2_wd001_mix01_128_0316
# CUDA_VISIBLE_DEVICES=0 python3 action_mylstm.py --nlayers=3 --my 0 1  --ups=5 --lr=0.001 --wd=0.001 --stride=10 --bl=20 --nhid=128 --sb=s10b20_3layer_z2_wd001_mix01_128_up5_0321

# rebuttal
CUDA_VISIBLE_DEVICES=0 python3 action_mylstm_cv.py  --nlayers=4 --data_mode=CV --lr=0.001 --cell_type=ELEATT_LSTM --nhid=512 --sb=cv_ELEATT_512_4layer_0618
