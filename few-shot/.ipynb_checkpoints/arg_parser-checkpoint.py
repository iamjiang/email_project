import os
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    parser.add_argument('--gpus', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--shuffle_train",  type=bool,default=True,help="shuffle data or not")
    parser.add_argument("--validation_split",  type=float,default=0.2,help="The split ratio for validation dataset")
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=2,help="Undersampling negative vs position ratio in training")
    # parser.add_argument("--test_negative_positive_ratio",  type=int,default=10,help="Undersampling negative vs position ratio in test set")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--truncation_strategy", type=str, default="head",help="how to truncate the long length email")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    #     parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--use_schedule', action="store_true")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--model_checkpoint', type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--model_output_name",  type=str, default=None)
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--frozen_layers", type=int, default=0,help="freeze layers without gradient updates")

    args= parser.parse_args()

    # args.model_output_name=f'{args.model_output_name}_{args.truncation_strategy}'
    # args.output_dir=f'{args.output_dir}_{args.truncation_strategy}'
    args.output_dir=args.model_checkpoint.split("-")[0] + "_" + args.model_checkpoint.split("-")[1] + "_repo"
    args.model_output_name=args.model_checkpoint.split("-")[0] + "_" + args.model_checkpoint.split("-")[1]

    return args

args=arg_parse()
# print()
# print(args)
# print()