CUDA_VISIBLE_DEVICES=0 python rqvae.py \
      --data_path ../data/Amazon18/Office/Office.emb-qwen-td.npy \
      --ckpt_dir ./output/Office \
      --num_emb_list 512 512 512 512 512 \
      --sk_epsilons 0.0 0.0 0.0 0.0 0.0 \
      --device cuda:0 \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 2048
