from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer
from genrec.models.DIFF_GRM.trainer import DIFF_GRMTrainer
from genrec.models.DIFF_GRM.evaluator import DIFF_GRMEvaluator
from genrec.models.DIFF_GRM.collate_block import block_masking
from genrec.models.DIFF_GRM.model_block import DIFF_GRM_BLOCK

# Aliases so get_tokenizer / get_trainer / get_evaluator can find them
tokenizer = DIFF_GRMTokenizer
trainer = DIFF_GRMTrainer
evaluator = DIFF_GRMEvaluator
