python main_segment.py \
--dataset 'snrINF' \
--data_root '../Data/subtomograms_snrINF/' \
--resume './outputs/2-way-1-shot_dusescnn_snrINF/model_25.pkl' \
--net 'dusescnn' \
--N 2 \
--K 1 \
--n_epochs 40 \
--batchs 1000 \
--batch_size 10 \
--lr 1e-4 \
--eval_epochs 5

