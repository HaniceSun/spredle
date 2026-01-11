import argparse
from importlib import resources
from .downloader import DataDownloader
from .preprocess import DataProcessor
from .dataset import CustomDataset
from .trainer import Trainer


def get_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', required=True)

    p1 = subparsers.add_parser("download-training-data", help="get training data from Ensembl")
    p1.add_argument('--genome_reference', type=str, default=None, help='downloading Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz from Ensembl if not specified')
    p1.add_argument('--gene_annotation', type=str, default=None, help='downloading Homo_sapiens.GRCh38.115.gtf.gz from Ensembl if not specified')

    p2 = subparsers.add_parser("preprocess", help="preprocess the downloaded training data")
    p2.add_argument('--input', type=str, default='Homo_sapiens.GRCh38.115_seq.txt', help='the input file')
    p2.add_argument('--nt', type=int, default=5000, help='length of the chunk sequences to extract')
    p2.add_argument('--flank', type=int, default=5000, help='length of flanking sequences to extract')

    p3 = subparsers.add_parser("torch-dataset", help="generate torch dataset from the processed files")
    p3.add_argument('--input', type=str, default='Homo_sapiens.GRCh38.115_seq_nt5000_flank5000.txt', help='input file of the processed data')
    p3.add_argument('--train_chroms', type=str, default=None, help='the chromosomes used as train dataset')
    p3.add_argument('--val_chroms', type=str, default=None, help='the chromosomes used as val dataset')

    p4 = subparsers.add_parser("train", help="train the deep learning models")
    p4.add_argument('--config_file', type=str, default='config.yaml', help='configuration file for model training')
    p4.add_argument('--model_name', type=str, default='SpliceAI', help='the model to train')
    p4.add_argument('--train_file', type=str, default='dataset_train.pt', help='training dataset file')
    p4.add_argument('--val_file', type=str, default='dataset_val.pt', help='validation dataset file')
    p4.add_argument('--metrics_file', type=str, default='metrics.txt', help='metrics output file')
    p4.add_argument('--lambda_lr', type=str, default=None, help='learning rate as a string seperated by comma for different epochs')

    p5 = subparsers.add_parser("test", help="test a trained model")
    p5.add_argument('--test_file', type=str, default='dataset_test.pt', help='test dataset file')
    p5.add_argument('--model_name', type=str, default='SpliceAI', help='the model to test')
    p5.add_argument('--epoch', type=int, default=4, help='the epoch of the trained model to load')

    p6 = subparsers.add_parser("predict", help="predict using a trained model")
    p6.add_argument('--pred_seq', type=str, default='ATCG,ATCG', help='sequences to predict, separated by comma')
    p6.add_argument('--pred_file', type=str, default='dataset_pred.pt', help='pred dataset file')
    p6.add_argument('--model_name', type=str, default='SpliceAI', help='the model to test')
    p6.add_argument('--epoch', type=int, default=4, help='the epoch of the trained model to load')
    p6.add_argument('--out_file', type=str, default='predicted.txt', help='prediction output file')

    p7 = subparsers.add_parser("saliency", help="generate saliency map using a trained model")
    p7.add_argument('--sal_seq', type=str, default='ATCG,ATCG', help='sequences to get saliency map, separated by comma')
    p7.add_argument('--out_file', type=str, default='saliency.txt', help='saliency output file')
    p7.add_argument('--model_name', type=str, default='SpliceAI', help='the model to test')
    p7.add_argument('--epoch', type=int, default=4, help='the epoch of the trained model to load')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.command == 'download-training-data':
        dd = DataDownloader(genome_reference=args.genome_reference, gene_annotation=args.gene_annotation)
        dd.download_training_data()
    if args.command == 'preprocess':
        dp = DataProcessor()
        dp.cut_seq(in_file=args.input, nt=args.nt, flank=args.flank)
    elif args.command == 'torch-dataset':
        train_chroms = args.train_chroms
        val_chroms = args.val_chroms
        if not train_chroms or not val_chroms:
            train_chroms = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21', 'chr2',
                            'chr4','chr6', 'chr8', 'chr10', 'chr12', 'chr14', 'chr16',
                            'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
            val_chroms = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']

        out_train = args.input.split('.txt')[0] + '_train.pt'
        out_val = args.input.split('.txt')[0] + '_val.pt'
        ds_train = CustomDataset(args.input, train_chroms)
        ds_val = CustomDataset(args.input, val_chroms)
        ds_train.split_and_save_dataset(out_train)
        ds_val.split_and_save_dataset(out_val)

    elif args.command == 'train':
        lambda_lr = None
        if args.lambda_lr:
            lambda_lr = [float(str(x)) for x in args.lambda_lr.split(',')]

        trainer = Trainer(config_file=args.config_file, model_name=args.model_name,
                          train_file=args.train_file, val_file=args.val_file,
                          metrics_file=args.metrics_file, lambda_lr=lambda_lr)
        trainer.count_parameters()
        trainer.run()
    elif args.command == 'test':
        trainer = Trainer(test_file=args.test_file)
        trainer.test(model_name=args.model_name, epoch=args.epoch)
    elif args.command == 'predict':
        trainer = Trainer()
        pred_seq = [s.strip() for s in args.pred_seq.split(',')]
        trainer.predict(model_name=args.model_name, epoch=args.epoch, pred_seq=pred_seq,
                        pred_file=args.pred_file, out_file=args.out_file)
    elif args.command == 'saliency':
        trainer = Trainer()
        sal_seq = [s.strip() for s in args.sal_seq.split(',')]
        trainer.saliency_map(model_name=args.model_name, epoch=args.epoch, sal_seq=sal_seq, out_file=args.out_file)


if __name__ == '__main__':
    main()
