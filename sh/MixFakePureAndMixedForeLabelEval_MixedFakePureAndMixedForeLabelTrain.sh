python main_SSL_LA_ddp.py \
--protocols_path /data/lqc/datasets/DetComplexAudio/new_protocols/new_MixFake_Mixed_and_Fore_ForeLabel.txt \
--eval_dataset MixFakePureAndMixedForeLabel \
--eval_output /path/to/Scores \
--model_path /path/to/ckpt  \
--comment MixedFake_PureAndMixedForeLabel_promptV4_baseline \
--gpus 2 \
--batch_size 32 \
--lr 0.005 \
--eval
