python main_SSL_LA_ddp.py \
--protocols_path /data/lqc/datasets/DetComplexAudio/new_protocols/new_MixFake_Mixed_and_Fore_ForeLabel.txt \
--batch_size 32 \
--num_epochs 30 \
--gpus 4 5 6 7  \
--lr 0.005 \
--algo 0 \
--comment MixedFake_PureAndMixedForeLabel
