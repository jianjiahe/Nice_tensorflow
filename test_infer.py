from infer import Infer
from config import ModelBasic,TrainBasic


infer = Infer(corpus_name=TrainBasic.dataset, run_name=TrainBasic.runname, sample_num=TrainBasic.batch_size, sample_dim=ModelBasic.in_out_dim)

infer.generate(0, 0)